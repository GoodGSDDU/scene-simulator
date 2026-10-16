function [cube, meta] = read_signal(folder, frameIndex)
%READ_SIGNAL 读取 2D 场景雷达仿真器导出的单帧数据。
%
%   [cube, meta] = read_signal(folder, frameIndex)
%
%   folder     : 数据集目录（含 signal.bin 与 meta.json）
%   frameIndex : 帧号，从 0 开始（默认 0）
%   cube       : (Nc, Nv, Ns) 复数矩阵，已乘以 1/iq_scale
%   meta       : 解析后的 meta.json 结构体
%
% 支持两种布局（见 docs/DESIGN.md 第 8.2 节）：
%   'ti_dca1000'    : 每 chirp 内“阵元在外、采样在内”，每 2 个复数打包成
%                     4 个 int16 —— [Re(v1), Re(v2), Im(v1), Im(v2), ...]，
%                     与 TI 官方 readDCA1000.m 一致
%   'interleaved_iq': 每采样直接 [Re, Im]
%
% ⚠️ MATLAB 是列主序、Python 是 C 序：同一份二进制在两边 reshape 的
%    维度顺序互为逆序，这是最容易写错的地方。

    if nargin < 2
        frameIndex = 0;
    end
    folder = char(folder);
    meta = jsondecode(fileread(fullfile(folder, 'meta.json')));

    shape = double(meta.signal.shape_per_frame(:)');   % [Nc, Nv, Ns]
    nc = shape(1); nv = shape(2); ns = shape(3);
    frameBytes = double(meta.signal.frame_bytes);
    layout = char(meta.signal.layout);

    signalName = 'signal.bin';
    if isfield(meta, 'signal_file')
        signalName = char(meta.signal_file);
    end

    fid = fopen(fullfile(folder, signalName), 'r');
    if fid < 0
        error('无法打开 %s', signalName);
    end
    % 直接定位到目标帧 —— 绝不要逐帧从头读（O(N^2) I/O）
    fseek(fid, frameIndex * frameBytes, 'bof');
    raw = fread(fid, frameBytes / 2, 'int16=>double');
    fclose(fid);
    if numel(raw) ~= frameBytes / 2
        error('signal.bin 长度不足，读取到 %d 个 int16', numel(raw));
    end

    flat = unpack_flat(raw, nc, nv, ns, layout);      % (nv*ns, nc)

    cube = complex(zeros(nc, nv, ns));
    for m = 1:nc
        % 行序为 [阵元1 的 Ns 个采样][阵元2 的 Ns 个采样]...
        cube(m, :, :) = reshape(flat(:, m), [ns, nv]).';
    end

    scales = double(meta.signal.iq_scale_per_frame(:));
    if isempty(scales)
        scales = double(meta.signal.iq_scale);
    end
    scale = scales(min(frameIndex + 1, numel(scales)));
    cube = cube / scale;
end

% ---------------------------------------------------------------------------
function v = unpack_flat(raw, nc, nv, ns, layout)
%UNPACK_FLAT 把 int16 流还原成每 chirp 一列、行序为“阵元外/采样内”的复矩阵。
    flatLen = nv * ns;
    blk = reshape(raw, [flatLen * 2, nc]);           % 列主序：每列一个 chirp

    if strcmp(layout, 'interleaved_iq')
        re = blk(1:2:end, :);
        im = blk(2:2:end, :);
        v = re + 1i * im;
        return;
    end
    if ~strcmp(layout, 'ti_dca1000')
        error('未知布局: %s', layout);
    end
    if mod(flatLen, 2) ~= 0
        error('ti_dca1000 布局要求 Nv*Ns 为偶数');
    end
    reA = blk(1:4:end, :);      % Re(v1), Re(v3), ...
    reB = blk(2:4:end, :);      % Re(v2), Re(v4), ...
    imA = blk(3:4:end, :);
    imB = blk(4:4:end, :);

    v = complex(zeros(flatLen, nc));
    v(1:2:end, :) = reA + 1i * imA;
    v(2:2:end, :) = reB + 1i * imB;
end
