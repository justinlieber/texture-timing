


function bestLags = GetCellLatencies(fullPsth)
% GetCellLatencies(fullPsth)
%   Takes in fullPsth, which is a matrix of PSTHs
%       Dim 1: time
%       Dim 2: cells
%       Dim 3: conditions
%       Dim 4: repetitions (averaged out)
%
% GetCellLatencies() loops through every pair of cells, and finds the
% cross-correlation of their PSTHs for each condition individually. These
% cross-correlations are then averaged across all conditions. The peak of
% this distribution is the optimal peak for the pair of cells. 
%
% The best peak is computed as an average over the population of pairs,
% with the average weighted by the strength of each pair's cross-correlation. 
%
% Peaks are reported in units of bins, and should be multiplied by the bin
% size


nCells = size(fullPsth,2);
nConditions = size(fullPsth,3);

saveAllXCorr = [];
for cInd = 1:nCells
    for cInd2 = 1:nCells
        theseCov = [];
        for tInd = 1:nConditions

            trace       = nanmean(nanmean(fullPsth(:,cInd,tInd,:),4),2);
            popTrace    = nanmean(nanmean(fullPsth(:,cInd2,tInd,:),4),2);
            thisCov     = xcov(trace, popTrace, 'coeff');

            if any(abs(thisCov)==Inf)
                thisCov(:) = NaN;
            end
            
            theseCov(:,tInd) = thisCov;
            
        end
        saveAllXCorr(:,cInd,cInd2) = nanmean(theseCov,2);
    end
end

[bestPairLagVal,  bestPairLagInd] = max(saveAllXCorr,[],1);

bestPairLagInd = bestPairLagInd - size(fullPsth,1);

for i=1:size(bestPairLagVal,2)
    bestPairLagVal(1,i,i) = NaN;
    bestPairLagInd(1,i,i) = NaN;
end

bestLag = squeeze(nansum(bestPairLagInd.*bestPairLagVal,2)./nansum(bestPairLagVal,2));

bestLag = bestLag - mean(bestLag);
