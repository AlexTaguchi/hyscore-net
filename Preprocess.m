% Process 14N Time-domain HYSCORE Spectra Function
function [xyF,specF] = Preprocess(filename)

% Load Phased Time Domain Spectrum
[xyT,specT] = eprload(strcat('Spectra/',filename));
if iscell(xyT)
    xyT = xyT{1,1}/1E9; % nanoseconds to seconds
else
    xyT = xyT/1E9; % nanoseconds to seconds
end
specT = real(specT); % remove imaginary

% Baseline Correction
yb = basecorr(real(specT),1,3);
yb = basecorr(yb,2,3);

% Window Function
% w = apowin('ham+',512);
% w2d = w(:)*w(:).';
% yb = yb.*w2d(1:256,1:256);

% Modulus of Fourier Transform
spec1 = abs(fftshift(fftn(yb,size(yb))));
stepF = 1E-6/((xyT(2)-xyT(1))*length(xyT));
lowF = -stepF*length(xyT)/2;
highF = stepF*(length(xyT)/2-1);
xyF = lowF:stepF:highF;

% Generate Fourier Noise Spectrum
specN = abs(fftshift(fftn(randn(length(yb)),size(yb))));
specN = sqrt(specN); % suppress noise spikes
specN = specN/mean(mean(specN)); % set mean to 1

% Select Peak Regions
% box corners defined by [x1, x2, y1, y2] (MHz)
bounds = [-20 20 0 20];
boxInd1 = zeros(4,1);
for n = 1:4
    [~,boxInd1(n)] = min(abs(xyF-bounds(n)));
end
spec2 = zeros(size(spec1));
spec2(boxInd1(1):boxInd1(2),boxInd1(3):boxInd1(4)) = ...
    spec1(boxInd1(1):boxInd1(2),boxInd1(3):boxInd1(4));
spec2(boxInd1(3):boxInd1(4),boxInd1(1):boxInd1(2)) = ...
    spec1(boxInd1(3):boxInd1(4),boxInd1(1):boxInd1(2));
boxInd2 = (length(xyF)/2)-(boxInd1-(length(xyF)/2)-1);
spec2(boxInd2(4):boxInd2(3),boxInd2(2):boxInd2(1)) = ...
    spec1(boxInd2(4):boxInd2(3),boxInd2(2):boxInd2(1));
spec2(boxInd2(2):boxInd2(1),boxInd2(4):boxInd2(3)) = ...
    spec1(boxInd2(2):boxInd2(1),boxInd2(4):boxInd2(3));

% Peak Region Removal
exInd = zeros(4,1);
for n = 1:4
    [~,exInd(n)] = min(abs(xyF)); 
end
for n = 1:2:4
    ex1 = exInd(n):exInd(n+1);
    ex2 = length(xyF)-exInd(n+1):length(xyF)-exInd(n);
    noise = median([spec2(ex1(1),ex1) spec2(ex1(end),ex1)...
                  spec2(ex1,ex1(1))' spec2(ex1,ex1(end))']);
    spec2(ex1,ex1) = noise*specN(ex1,ex1); % upper right region
    spec2(ex2,ex2) = noise*specN(ex2,ex2); % lower left region
end

% Interpolation
[xm,ym] = meshgrid(linspace(-12.8,12.8,257));
spec3 = interp2(xyF,xyF,spec2,xm,ym,'cubic');

% Symmetrize
spec3 = spec3+spec3';

% Convert to (+/+) Square Representation
spec3 = tril(spec3); % zero triangular half
spec3 = rot90(triu(rot90(spec3),1),-1); % zero triangular half
spec3 = spec3 + rot90(spec3); % create desired (+/+) quadrant
spec3 = spec3((ceil(end/2)+1):end,(ceil(end/2)+1):end); % keep (+/+)
spec3 = spec3/max(max(spec3)); % normalize to max height 1

% Output Symmetrized Square Matrix
xyF = xm(1,(uint16(end/2)+1):end);
specF = reshape(spec3,1,[]);

end