function [di, ki] = pick_peaks(power, mask, guard, maxPeaks)
%PICK_PEAKS  CFAR 命中点上的贪心非极大抑制，避免把同一个目标报成几十个点。
%
%   [di, ki] = pick_peaks(power, mask, guard, maxPeaks)
%
% 返回按功率降序排列的 (行索引, 列索引)；在本项目里行 = 多普勒、列 = 距离。

    if nargin < 3 || isempty(guard), guard = [2 2]; end
    if nargin < 4 || isempty(maxPeaks), maxPeaks = 512; end

    [rows, cols] = find(mask);
    if isempty(rows)
        di = zeros(0, 1); ki = zeros(0, 1); return;
    end
    values = power(sub2ind(size(power), rows, cols));
    [~, order] = sort(values, 'descend');
    rows = rows(order); cols = cols(order);

    keptR = zeros(0, 1); keptC = zeros(0, 1);
    for n = 1:numel(rows)
        if isempty(keptR) || ...
           all(abs(rows(n) - keptR) > guard(1) | abs(cols(n) - keptC) > guard(2))
            keptR(end+1) = rows(n); %#ok<AGROW>
            keptC(end+1) = cols(n); %#ok<AGROW>
            if numel(keptR) >= maxPeaks
                break;
            end
        end
    end
    di = keptR; ki = keptC;
end
