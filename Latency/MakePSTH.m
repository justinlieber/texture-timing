
function fullPsth = MakePSTH(spikeCells, psthOptions)

% cortical values
minVal      = psthOptions.minT;
maxVal      = psthOptions.maxT;
tLength     = maxVal-minVal;

sigmaVal    = psthOptions.tempRes;
binSize     = psthOptions.binSize;

binEdges    = 0:binSize:tLength;
binCenters  = binEdges(1:end-1) + binSize/2;


% Correcting for edge effects:
% Assume that we're in a steady-state. That means there are some spikes
% before minVal and after maxVal that we're missing when we do our Gaussian
% convolution. If we don't model these, we're have artificially low spike
% rates on each edge (the effects of which will be worst for large values
% of sigma.

% To correct for these, we assume that, on average, we can take the firing
% rate of this particular trial and extrapolate it outside the bounds of
% minVal and maxVal. We treat the area outside the edges as a smooth step
% function firing at this mean firing rate.

% When we convolve a step function with a Gaussian, we get the function:
%    ( (erf(t/sigma) + 1)/2 )
% The integral of this function is (one for each side): 
func1 = @(t,sigma)((t + -t.*erf(t/sigma) + -sigma*exp(-(t/sigma).^2)/sqrt(pi))/2);
func2 = @(t,sigma)((t + t.*erf(t/sigma) + sigma*exp(-(t/sigma).^2)/sqrt(pi))/2);

% For each bin, we take the diff of this function across the two bin edges
% (similar to what we do with the Gaussian below).
sideShape = (diff(func1(binEdges,sigmaVal) + func2(binEdges-(maxVal-minVal), sigmaVal)))'/binSize;


cutSpikes = cellfun(@(x)(x(x>minVal & x<maxVal)-minVal), spikeCells, 'UniformOutput', false');

N = size(spikeCells);
if length(N) < 5
    N = [N ones(1,5-length(N))];
end

fullPsth = nan([length(binCenters) N]);
getRates = [];
for cInd = 1:N(1)
    for tInd = 1:N(2)
        for rInd = 1:N(3)
            for otherInd = 1:N(4)
                spikes = cutSpikes{cInd,tInd,rInd,otherInd};
                
                if ~isvector(spikes)
                    continue;
                end
                
                fullPsth(:,cInd,tInd,rInd,otherInd) = 0;
                
                % fill in the sides
                getRates(cInd,tInd,rInd,otherInd) = length(spikes)/(maxVal-minVal);
                thisShape = sideShape.*getRates(cInd,tInd,rInd,otherInd)*(binSize);
                fullPsth(:,cInd,tInd,rInd,otherInd) = fullPsth(:,cInd,tInd,rInd,otherInd) + thisShape;
                
                % convolve with the Gaussian
                for spInd = 1:length(spikes)
                    fullPsth(:,cInd,tInd,rInd,otherInd) = fullPsth(:,cInd,tInd,rInd,otherInd) + ...
                        diff(normcdf(binEdges, spikes(spInd),sigmaVal))';
                end
            end
        end
    end
    cInd
end
