%% demo.m —— 2D 场景雷达仿真器数据处理完整链路（MATLAB）
%
% 用法：先改 DATASET 指向导出目录，或在命令行运行
%   demo('../../dataset/demo')
%
% 处理链：读取二进制 → 距离 FFT → 多普勒 FFT → CA-CFAR → 阵列 FFT 测角 → 点云
% 所有公式与 templates/python 保持一致（docs/DESIGN.md 第 10.6 节）。

function demo(datasetFolder)
    if nargin < 1 || isempty(datasetFolder)
        datasetFolder = fullfile('..', '..', 'dataset', 'demo');
    end
    frameIndex = 0;

    %% 1. 读取
    [cube, meta] = read_signal(datasetFolder, frameIndex);
    [nc, nv, ns] = size(cube);

    fs    = double(meta.radar.sample_rate_hz);
    S     = double(meta.radar.slope_hz_per_s);
    Tr    = double(meta.radar.ramp_time_s);
    Td    = double(meta.radar.idle_time_s);
    Tc    = Tr + Td;
    c     = 299792458;
    lam   = c / double(meta.radar.center_freq_hz);
    d     = double(meta.radar.element_spacing_m);

    % 派生指标（务必用有效带宽，而不是整段扫频带宽 S*Tr）
    rangeBin = c * fs / (2 * S * ns);
    Rmax     = ns * rangeBin;
    vRes     = lam / (2 * nc * Tc);
    vMax     = lam / (4 * Tc);

    fprintf('数据: %dx%dx%d (Nc,Nv,Ns),  布局 %s\n', nc, nv, ns, meta.signal.layout);
    fprintf('ΔR = %.4f m   Rmax = %.2f m   Δv = %.4f m/s   vmax = ±%.3f m/s\n', ...
            rangeBin, Rmax, vRes, vMax);

    %% 2. 距离 FFT（周期窗，不是对称窗）
    rangeAxis = (0:ns-1).' * rangeBin;
    rangeWin = hann_periodic(ns);
    afterRange = fft(cube .* reshape(rangeWin, 1, 1, []), ns, 3);
    % 复采样下全部 Ns 个 bin 都可用（不需要只取一半）

    profile = squeeze(sqrt(sum(sum(abs(afterRange).^2, 1), 2)));

    %% 3. 多普勒 FFT
    dopWin = hann_periodic(nc);
    afterDoppler = fftshift(fft(afterRange .* reshape(dopWin, [], 1, 1), nc, 1), 1);
    rdMap = squeeze(sum(abs(afterDoppler).^2, 2));                % (nc, ns)
    rdDb  = 10 * log10(max(rdMap, 1e-30));

    % 速度轴：绝不能用 linspace(-vmax, vmax, nc)
    vAxis = ((0:nc-1).' - nc / 2) * vRes;

    %% 4. CA-CFAR
    pfa = 1e-3;
    guard = [2 2]; train = [4 4];
    [mask, alpha] = ca_cfar_2d(rdMap, guard, train, pfa);
    [di, ki] = pick_peaks(rdMap, mask, guard, 100);
    fprintf('CFAR: alpha = %.3f, 抑制后检测数 = %d\n', alpha, numel(di));

    %% 5. 阵列 FFT 测角（快拍取自复数距离-多普勒立方体）
    nFft = 180;
    angleAxis = asin(max(min(((0:nFft-1).' - nFft/2) / nFft * lam / d, 1), -1));
    winEl = hann_periodic(nv);
    azimuths = zeros(numel(di), 1);
    for n = 1:numel(di)
        snap = squeeze(afterDoppler(di(n), :, ki(n))) .* winEl;
        spectrum = abs(fftshift(fft(snap, nFft))).^2;
        [~, best] = max(spectrum);
        azimuths(n) = angleAxis(best);
    end

    ranges = (ki(:) - 1) * rangeBin;
    speeds = vAxis(di(:));
    fprintf('\n%8s %12s %10s\n', '距离(m)', '速度(m/s)', '方位(°)');
    for n = 1:numel(di)
        fprintf('%8.2f %+12.3f %10.2f\n', ranges(n), speeds(n), rad2deg(azimuths(n)));
    end

    %% 6. 出图
    figure('Name', '2D Radar Scene Simulator - MATLAB demo', 'Color', 'w');

    subplot(2, 2, 1);
    plot(real(cube(1, 1, 1:min(128, ns))), 'b'); hold on;
    plot(imag(cube(1, 1, 1:min(128, ns))), 'r');
    title('原始 IQ（chirp0 / 阵元0）'); xlabel('采样点'); legend('I', 'Q'); grid on;

    subplot(2, 2, 2);
    plot(rangeAxis, 20 * log10(max(profile, 1e-30)));
    title('距离像'); xlabel('距离 (m)'); ylabel('dB'); xlim([0 Rmax]); grid on;

    subplot(2, 2, 3);
    imagesc(rangeAxis, vAxis, rdDb);
    axis xy; colormap(gca, 'turbo'); colorbar;
    hold on; plot(ranges, speeds, 'ro', 'MarkerSize', 8, 'LineWidth', 1.2);
    title('距离-多普勒（红圈 = CFAR 检测）');
    xlabel('距离 (m)'); ylabel('速度 (m/s)');

    subplot(2, 2, 4);
    if ~isempty(di)
        xy = [ranges .* cos(azimuths), ranges .* sin(azimuths)];
        scatter(xy(:, 1), xy(:, 2), 40, 'filled'); hold on;
    end
    plot(0, 0, '^', 'MarkerSize', 10, 'LineWidth', 2);
    axis equal; grid on;
    title('点云（雷达坐标系）'); xlabel('前方 X (m)'); ylabel('横向 Y (m)');
end

% ---------------------------------------------------------------------------
function w = hann_periodic(n)
%HANN_PERIODIC 周期（DFT-even）窗 —— hann(n) 是对称窗，做 FFT 处理不合适。
    w = 0.5 - 0.5 * cos(2 * pi * (0:n-1).' / n);
end
