function [mask, alpha, threshold] = ca_cfar_2d(power, guard, train, pfa)
%CA_CFAR_2D  二维单元平均 CFAR。
%
%   [mask, alpha, threshold] = ca_cfar_2d(power, guard, train, pfa)
%
%   power  : (Nr, Nd) 功率图
%   guard  : [行, 列] 保护单元数
%   train  : [行, 列] 训练单元数（不含保护单元）
%   pfa    : 期望虚警概率
%
%   阈值 = alpha * 参考单元功率均值，其中
%       alpha = Nref * (pfa^(-1/Nref) - 1)
%   这样指定的虚警率才有意义；用 mean + k*sigma 是无法控制虚警率的。

    if nargin < 2 || isempty(guard), guard = [2 2]; end
    if nargin < 3 || isempty(train), train = [4 4]; end
    if nargin < 4 || isempty(pfa), pfa = 1e-3; end

    [nr, nd] = size(power);
    gh = guard(1); gw = guard(2);
    th = train(1); tw = train(2);

    nOuter = (2 * (gh + th) + 1) * (2 * (gw + tw) + 1);
    nInner = (2 * gh + 1) * (2 * gw + 1);
    nRef = nOuter - nInner;
    alpha = nRef * (pfa ^ (-1 / nRef) - 1);

    mask = false(nr, nd);
    threshold = zeros(nr, nd);

    for i = (gh + th + 1):(nr - gh - th)
        r0 = i - gh - th; r1 = i + gh + th;
        for j = (gw + tw + 1):(nd - gw - tw)
            c0 = j - gw - tw; c1 = j + gw + tw;
            outer = sum(sum(power(r0:r1, c0:c1)));
            inner = sum(sum(power(i-gh:i+gh, j-gw:j+gw)));
            noise = max(outer - inner, 0) / nRef;
            threshold(i, j) = alpha * noise;
            mask(i, j) = power(i, j) > threshold(i, j) && noise > 0;
        end
    end
end
