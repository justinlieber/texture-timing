

%% Get the PSTH

psthOptions         = [];
psthOptions.minT    = 0.2;
psthOptions.maxT    = 1.8;
psthOptions.tempRes = 5/1000;
psthOptions.binSize = 1/1000;


fullPsth = MakePSTH(cdaData.fullRun.data, psthOptions);

%% Get latencies

bestIndLag = GetCellLatencies(fullPsth);

%% Plot lags

bestIndShift = round(bestIndLag);
maxShift = max(abs(bestIndShift));

%bestIndShift = zeros(141,1);
%maxShift = 160;


tInd = 51;

noShiftIm   = nan([size(fullPsth,1)+maxShift.*2 size(fullPsth,2)]);
baseIm      = nan([size(fullPsth,1)+maxShift.*2 size(fullPsth,2)]);
traceVar = [];
for cInd=1:141
    indVals = (1:size(fullPsth,1))+maxShift+bestIndShift(cInd);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4)./mean(thisTrace);
    traceVar(cInd) = nanstd(thisTrace);
    
    
    noShiftIm((1:size(fullPsth,1))+maxShift,cInd) = thisTrace;
    baseIm(indVals,cInd) = thisTrace;
end

[~,cOrder] = sort(traceVar,'descend');

clf;

subplot(2,1,1);
imagesc(noShiftIm(:,cOrder)');
caxis([0 nanmedian(baseIm(:))+mad(baseIm(:),1)*3]);
title(cdaData.fullRun.textures{tInd});

subplot(2,1,2);
imagesc(baseIm(:,cOrder)');
caxis([0 nanmedian(baseIm(:))+mad(baseIm(:),1)*3]);
title(cdaData.fullRun.textures{tInd});

