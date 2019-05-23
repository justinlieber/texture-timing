
%%
tmr = [];

for typeInd = 1:3
    thisCList = periphData.typeInd{typeInd};
    tmr(typeInd,:) = squeeze( nanmean(nanmean(periphData.rates(2,thisCList,:,:),4),2) );
end


clf;
subplot(1,2,1)
hold all;
sh = scatter(tmr(3,:), tmr(2,:), 'MarkerEdgeColor', 'k','LineWidth', 1.4);
th = text(75*(3/5), 125/5, sprintf('r=%0.2f', nancorr(tmr(3,:), tmr(2,:))));
set(th, 'FontSize', 12);
axis square;
axis([0 75 0 125]);
chamod();

xlabel('SA1 rate (Hz)');
ylabel('RA rate (Hz)');


subplot(1,2,2);
scatter(tmr(2,:), tmr(1,:), 'MarkerEdgeColor', 'k','LineWidth', 1.4);
th = text(125*(3/5), 165/5, sprintf('r=%0.2f', nancorr(tmr(2,:), tmr(1,:)) ));
set(th, 'FontSize', 12);
axis square;
axis([0 125 0 165]);
chamod();

xlabel('RA rate (Hz)');
ylabel('PC rate (Hz)');

% 
export_fig -m2 AfferentTypeTextureResponse.png


%% Data

size( cdaData.fullRun.data )

%% How do we make PSTHs for these? What resolution?

minVal = 0.2;
maxVal = 1.8;
tLength = maxVal-minVal;

sigmaVal = 0.005;

binSize = 1/1000;

binEdges = 0:binSize:tLength;
binCenters = binEdges(1:end-1) + binSize/2;

% Integral of ( (erf(t) + 1)/2 ), which is a good way to correct for edge
% effects - we treat the sides as a step function set to the average firing 
% rate over the course of the trial.
func1 = @(t,sigma)((t + -t.*erf(t/sigma) + -sigma*exp(-(t/sigma).^2)/sqrt(pi))/2);
func2 = @(t,sigma)((t + t.*erf(t/sigma) + sigma*exp(-(t/sigma).^2)/sqrt(pi))/2);


sideShape = (diff(func1(binEdges,sigmaVal) + func2(binEdges-(maxVal-minVal), sigmaVal)))'/binSize;


cutSpikes = cellfun(@(x)(x(x>minVal & x<maxVal)-minVal), cdaData.fullRun.data, 'UniformOutput', false');

getFullPsth = nan(length(binCenters),141,59,5);
getRates = [];
for cInd = 1:141
    getFullPsth(:,cInd,:,:,:) = 0;
    for tInd = 1:59
        for rInd = 1:5
            spikes = cutSpikes{cInd,tInd,rInd};
            getRates(cInd,tInd,rInd) = length(spikes)/(maxVal-minVal);
            thisShape = sideShape.*getRates(cInd,tInd,rInd)*(binSize);
            getFullPsth(:,cInd,tInd,rInd) = getFullPsth(:,cInd,tInd,rInd) + thisShape;
            for spInd = 1:length(spikes)
                getFullPsth(:,cInd,tInd,rInd) = getFullPsth(:,cInd,tInd,rInd) + ...
                    diff(normcdf(binEdges, spikes(spInd),sigmaVal))';;
            end
        end
    end
    cInd
end
fullPsth = getFullPsth;

clear getFullPsth;

'got psth'


%% Plot Traces

%cInd = 2;
tInd = 45;


%[sum(trace)/(maxVal-minVal) nanmean(getRates(cInd,tInd,:)) nanmean(cdaAnalysis.fullRate(cInd,tInd,:))]

clf;
hold all;
plot(binCenters, trace/binSize)
chamod();
title(cdaData.fullRun.textures{tInd});
xlim([0 1.6]);
%ylim([0 0.11]);


%% What about just basic stats?

% CV for each thing.
meanRate = nanmean(getRates,3);
traceCV = nanstd(nanmean(fullPsth,4),[],1)./(nanmean(nanmean(fullPsth,4),1)+eps);

%%

clf;
imagesc(squeeze(traceCV))
caxis([0 1]);

%%

scatter(meanRate(:), traceCV(:))

%% Find the best latency for each neuron

saveAllXCorr = [];
saveOffCorr = [];
for cInd = 1:141
    for cInd2 = 1:141
        theseCov = [];
        theseCov2 = [];
        for tInd = 1:59

            trace       = nanmean(nanmean(fullPsth(:,cInd,tInd,:),4),2);
            popTrace    = nanmean(nanmean(fullPsth(:,cInd2,tInd,:),4),2);
            thisCov     = xcov(trace, popTrace, 'coeff');

            if any(abs(thisCov)==Inf)
                thisCov(:) = NaN;
            end
            
            theseCov(:,tInd) = thisCov;

            
%             thisCov     = xcov(trace(end:-1:1), popTrace, 'coeff');
%             theseCov2(:,tInd) = thisCov;
            
            
        end
        saveAllXCorr(:,cInd,cInd2) = nanmean(theseCov,2);
        %saveAllXCorr(:,cInd2,cInd) = nanmean(theseCov,2);
        
        %saveOffCorr(:,cInd,cInd2) = nanmean(theseCov2,2);
        %saveOffCorr(:,cInd2,cInd) = nanmean(theseCov2,2);
    end
    cInd
end

%%


clf;

subplot(1,3,1);
[maxVal,  maxInd] = max(saveAllXCorr,[],1);
im = squeeze( maxVal);
for i=1:141
    im(i,i) = NaN;
end

imagesc( im )
axis image;
caxis([0 0.5]);

tMax = (size(fullPsth,1)-1)*binSize;
tVec = -tMax:binSize:tMax;

subplot(1,3,2);
im = squeeze( tVec(maxInd));
for i=1:141
    im(i,i) = NaN;
end

imagesc( im )
axis image;
caxis([-1 1]*0.1);


bestLag = squeeze(nansum(tVec(maxInd).*maxVal,2)./sum(maxVal,2));


subplot(1,3,3);
im = bestLag - bestLag';
for i=1:141
    im(i,i) = NaN;
end

imagesc( im )
axis image;
caxis([-1 1]*0.1);

%%
y = squeeze( tVec(maxInd));
x = bestLag' - bestLag;

clf
scatter(x(:), y(:));
nancorr(x(:),y(:))

axis([-1 1 -1 1]*1.5);
axis square;

%%
im = squeeze( abs(tVec(maxInd).*maxVal) );

[Y,e] = cmdscale(im);


subplot(1,2,1);
imagesc(im);
axis image;

subplot(1,2,2)
imagesc(Y(:,1)-Y(:,1)');
axis image;

%%
y = squeeze( tVec(maxInd) );
x = Y(:,1)'-Y(:,1);

clf
scatter(x(:), y(:));
nancorr(x(:),y(:))


%axis([-1 1 -1 1]*1.5);
axis square;

%%

bestIndShift = round((bestLag-mean(bestLag)).*1000);
nanmean(bestIndShift);
maxShift = max(abs(bestIndShift));

%bestIndShift = zeros(141,1);
maxShift = 160;


tInd = 44;

baseIm = nan([size(fullPsth,1)+maxShift.*2 size(fullPsth,2)]);
traceVar = [];
for cInd=1:141
    indVals = (1:size(fullPsth,1))+maxShift+bestIndShift(cInd);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4)./mean(thisTrace);
    traceVar(cInd) = nanstd(thisTrace);
    baseIm(indVals,cInd) = thisTrace;
end

[~,cOrder] = sort(traceVar,'descend');

clf;
imagesc(baseIm(:,cOrder)');
caxis([0 nanmedian(baseIm(:))+mad(baseIm(:),1)*3]);
title(cdaData.fullRun.textures{tInd});


%% Run this same thing on neurons from a single hemisphere

hemCodes = unique(cdaData.neuronMetadata.hemCode);
cList = find(strcmp(cdaData.neuronMetadata.hemCode, hemCodes{3}));
%cList = 1:141;


tInd = 14;
[maxVal,  maxInd] = max(saveAllXCorr,[],1);


subBestLag = squeeze(nansum(tVec(maxInd(1,cList,cList)).*maxVal(1,cList,cList),2)./sum(maxVal(1,cList,cList),2));



bestIndShift = round((subBestLag-mean(subBestLag)).*1000);
nanmean(bestIndShift);
maxShift = max(abs(bestIndShift));

baseIm = nan([size(fullPsth,1)+maxShift.*2 size(fullPsth,2)]);
traceVar = [];
for cListInd=1:length(cList)
    cInd = cList(cListInd);
    indVals = (1:size(fullPsth,1))+maxShift+bestIndShift(cListInd);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4)./mean(thisTrace);
    traceVar(cListInd) = nanstd(thisTrace);
    baseIm(indVals,cListInd) = thisTrace;
end

[~,cOrder] = sort(traceVar,'descend');

clf;
imagesc(baseIm(:,cOrder)');
caxis([0 nanmedian(baseIm(:))+mad(baseIm(:),1)*3]);
title(cdaData.fullRun.textures{tInd});

%% New aproach

% First, find best lag based on full xcorr
% Second, iterate on xcorr between ind. and pop average


[maxVal,  maxInd] = max(saveAllXCorr,[],1);


hemCodes = unique(cdaData.neuronMetadata.hemCode);
cList = find(strcmp(cdaData.neuronMetadata.hemCode, hemCodes{3}));
cList = 1:141;


subBestLag = squeeze(nansum(tVec(maxInd(1,cList,cList)).*maxVal(1,cList,cList),2)./sum(maxVal(1,cList,cList),2));
bestIndShift = round((subBestLag-mean(subBestLag)).*1000);

nanmean(bestIndShift);
maxShift = max(abs(bestIndShift));

baseIm = zeros([size(fullPsth,1)+maxShift.*2 size(fullPsth,2) size(fullPsth,3)]);
for cListInd=1:length(cList)
    cInd = cList(cListInd);
    indVals = (1:size(fullPsth,1))+maxShift+bestIndShift(cListInd);
    allTraces = nanmean(fullPsth(:,cInd,:,:),4);
    allTraces = nanmean(fullPsth(:,cInd,:,:),4)./nanmean(allTraces,1);
    baseIm(indVals,cListInd,:) = allTraces;
end


centerVals = ((1:(2*size(baseIm,1)+1) ) - size(baseIm,1)-1 )*binSize;

textShift = [];
fullShift = [];
for cInd = 1:141
    mask = ones(141,1) == 1;
    mask(cInd) = false;
    
    popTraces = nanmean(baseIm(:,mask,:),2);
    cellTraces = baseIm(:,cInd,:);
    thisCov = [];
    for tInd = 1:59
        thisCov(:,tInd)     = xcov(cellTraces(:,1,tInd), popTraces(:,1,tInd), 'coeff');
        
        [~,ind] = max(nanmean(thisCov(:,tInd),2));
        textShift(cInd,tInd) = centerVals(ind);
    end
    
    [~,ind] = max(nanmean(thisCov,2));
    fullShift(cInd) = centerVals(ind);
end

newShift = newShift';

%%

tInd = 25;


%bestIndShift = round((subBestLag-nanmean(subBestLag) + textShift(tInd,:) - nanmean(textShift(tInd,:))).*1000);
%bestIndShift = round((subBestLag-nanmean(subBestLag) + fullShift(:) - nanmean(fullShift(:))).*1000);
bestIndShift = round((subBestLag-nanmean(subBestLag) ).*1000);

nanmean(bestIndShift);
maxShift = max(abs(bestIndShift));

maxShift = 350;

baseIm = nan([size(fullPsth,1)+maxShift.*2 size(fullPsth,2)]);
traceVar = [];
for cListInd=1:length(cList)
    cInd = cList(cListInd);
    indVals = (1:size(fullPsth,1))+maxShift+bestIndShift(cListInd);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4);
    thisTrace = nanmean(fullPsth(:,cInd,tInd,:),4)./mean(thisTrace);
    traceVar(cListInd) = nanstd(thisTrace);
    baseIm(indVals,cListInd) = thisTrace;
end

%[~,cOrder] = sort(traceVar,'descend');


clf;
imagesc(baseIm(:,cOrder)');
caxis([0 nanmedian(baseIm(:))+mad(baseIm(:),1)*3]);
title(cdaData.fullRun.textures{tInd});


newShift;


%%

N = (size(saveAllXCorr,1)-1)/2;
binLocs = ((-N:N)*binSize)';


maxCorrList= cell(141,1);
minCorrList= cell(141,1);
for cInd = 1:141
    mask = ones(141,1)==1;
    mask(cInd) = false;
    
    trace = nanmean(saveAllXCorr(:,cInd,mask),3);
    
    maxInd = find(diff(sign(diff(trace))) < 0)+1;
    [~, order] = sort(trace(maxInd), 'descend');
    
    maxCorrList{cInd} = [maxInd(order) binLocs(maxInd(order)) trace(maxInd(order))];
    
    
    minInd = find(diff(sign(diff(trace))) > 0)+1;
    [~, order] = sort(trace(minInd), 'ascend');
    minCorrList{cInd} = [minInd(order) binLocs(minInd(order)) trace(minInd(order))];
end

%%
%clf;
%hold all;
indList = [];
for cInd = 1:141
    
    corr = maxCorrList{cInd}(:,3);
    ind = find(corr./corr(1) > 0.9);
    if(length(ind)==1)
        indList(cInd) = maxCorrList{cInd}(ind,1);
    else
        [~,subInd] = min(abs(maxCorrList{cInd}(ind,2)'));
        [cInd ind(subInd)];
        
        indList(cInd) = maxCorrList{cInd}(ind(subInd),1);
%         
%         [cInd maxCorrList{cInd}(ind,2)']
%         
%         minCorr = minCorrList{cInd}(:,3);
%         minInd = find(-minCorr./corr(1) > 0.9);
%         
%         if ~isempty(minInd)
%             [cInd minCorrList{cInd}(minInd,2)']
%         end
        
    end
    
end

%%





%%

clf;
cInd = 52;
mask = ones(141,1)==1;
mask(cInd) = false;

plot( nanmean(saveAllXCorr(:,cInd,mask),3))

%%

thisInd = [];
for cInd = 1:141
    mask = ones(141,1)==1;
    mask(cInd) = false;
    
    trace = nanmean(saveAllXCorr(:,cInd,mask),3);
    [~,thisInd(cInd)] = max(trace);
end
%%

[~,ind] = sort(thisInd)




%%
usePsth = fullPsth;
firstLag = zeros(1,141);
runningLag = firstLag;

savePsthCrossCorr = [];
for cInd = 1:141
    cMask = ones(141,1)==1;
    cMask(cInd) = false;


    for tInd = 1:59

        trace       = nanmean(nanmean(usePsth(:,cInd,tInd,:),4),2);
        popTrace    = nanmean(nanmean(usePsth(:,cMask,tInd,:),4),2);
        thisCov     = xcov(trace, popTrace, 'coeff');

        savePsthCrossCorr(:,cInd,tInd) = thisCov;

        [~,ind] = max(thisCov);
        saveInd(cInd,tInd) = ind;
    end
    cInd
end

firstPsthCrossCorr = savePsthCrossCorr;

meanCorr = nanmean(firstPsthCrossCorr,3);
[~,lagInd] = max(meanCorr);
relLag = max(lagInd) - lagInd;

saveFirstLag = relLag;
runningLag = runningLag + relLag;

shiftPsth = nan([size(fullPsth) + [max(runningLag) 0 0 0]]);

for cInd = 1:141
    useInd = [1:size(fullPsth,1)]+runningLag(cInd);
    shiftPsth(useInd,cInd,:,:) = fullPsth(:,cInd,:,:);
end

mask = ~any(any(any(isnan(shiftPsth),2),3),4);
usePsth = shiftPsth(mask,:,:,:);
if mod(size(usePsth,1),2)==1
    usePsth = usePsth (1:end-1,:,:,:);
end

firstPsth = usePsth;



savePsthCrossCorr = [];
for cInd = 1:141
    cMask = ones(141,1)==1;
    cMask(cInd) = false;


    for tInd = 1:59

        trace       = nanmean(nanmean(usePsth(:,cInd,tInd,:),4),2);
        popTrace    = nanmean(nanmean(usePsth(:,cMask,tInd,:),4),2);
        thisCov     = xcov(trace, popTrace, 'coeff');

        if any(abs(thisCov)==Inf)
            thisCov(:) = NaN;
        end
        
        savePsthCrossCorr(:,cInd,tInd) = thisCov;

        [~,ind] = max(thisCov);
        saveInd(cInd,tInd) = ind;
    end
    cInd
end

secondPsthCrossCorr = savePsthCrossCorr;


meanCorr = nanmean(secondPsthCrossCorr,3);
[~,lagInd] = max(meanCorr);
relLag = max(lagInd) - lagInd;
runningLag = runningLag + relLag;

saveSecondLag = runningLag;

shiftPsth = nan([size(fullPsth) + [max(runningLag) 0 0 0]]);

for cInd = 1:141
    useInd = [1:size(fullPsth,1)]+runningLag(cInd);
    shiftPsth(useInd,cInd,:,:) = fullPsth(:,cInd,:,:);
end

mask = ~any(any(any(isnan(shiftPsth),2),3),4);
usePsth = shiftPsth(mask,:,:,:);
if mod(size(usePsth,1),2)==1
    usePsth = usePsth (1:end-1,:,:,:);
end

%%

usePsth = fullPsth;
firstLag = zeros(1,141);
runningLag = firstLag;

eachCrossCorr = cell(4,1);

for repInd = 1:4
    savePsthCrossCorr = [];
    for cInd = 1:141
        cMask = ones(141,1)==1;
        cMask(cInd) = false;


        for tInd = 1:59

            trace       = nanmean(nanmean(usePsth(:,cInd,tInd,:),4),2);
            popTrace    = nanmean(nanmean(usePsth(:,cMask,tInd,:),4),2);
            thisCov     = xcov(trace, popTrace, 'coeff');

            if any(abs(thisCov)==Inf)
                thisCov(:) = NaN;
            end

            savePsthCrossCorr(:,cInd,tInd) = thisCov;

            [~,ind] = max(thisCov);
            saveInd(cInd,tInd) = ind;
        end
        [repInd cInd]
    end

    eachCrossCorr{repInd} = savePsthCrossCorr;
    
    %secondPsthCrossCorr = savePsthCrossCorr;


    meanCorr = nanmean(savePsthCrossCorr,3);
    [~,lagInd] = max(meanCorr);
    relLag = max(lagInd) - lagInd;
    runningLag = runningLag + relLag;
    runningLag = runningLag - min(runningLag);

    %saveSecondLag = runningLag;

    shiftPsth = nan([size(fullPsth) + [max(runningLag) 0 0 0]]);

    for cInd = 1:141
        useInd = [1:size(fullPsth,1)]+runningLag(cInd);
        shiftPsth(useInd,cInd,:,:) = fullPsth(:,cInd,:,:);
    end

    mask = ~any(any(any(isnan(shiftPsth),2),3),4);
    usePsth = shiftPsth(mask,:,:,:);
    if mod(size(usePsth,1),2)==1
        usePsth = usePsth (1:end-1,:,:,:);
    end
end

%%

for i=1:4
    subplot(4,1,i);
    hold all;
    N = (size(eachCrossCorr{i},1)-1)/2;
    x = (-N:N)*binSize;
    for cInd = 1:141
        y = nanmean(eachCrossCorr{i}(:,cInd,:),3);
        plot(x, y./(max(y)+eps))
    end
end


%%

clf; 

N = (size(firstPsthCrossCorr(:,cInd,:),1)-1)/2;
x = (-N:N)*binSize;
subplot(2,1,1);
hold all;
for cInd = [1:141]+0
    y = nanmean(firstPsthCrossCorr(:,cInd,:),3);
    plot(x, y./max(y))
end

xlim([-2 2]);
chamod();

N = (size(secondPsthCrossCorr(:,cInd,:),1)-1)/2;
x = (-N:N)*binSize;
subplot(2,1,2);
hold all;
for cInd = 1:141
    y = nanmean(secondPsthCrossCorr(:,cInd,:),3);
    plot(x,y./max(y))
end

cInd = 127;
y = nanmean(secondPsthCrossCorr(:,cInd,:),3);
plot(x,y./(max(y)+eps), 'Color', 'k', 'LineWidth', 3)

xlim([-2 2]);
chamod();




%%

cInd = 1;

meanCorr = nanmean(savePsthCrossCorr,3);
N = length(trace)-1;



lagVals = (-N:N)*binSize;

clf;
plot(lagVals, meanCorr(:,cInd));

%%

%% Alignment

meanCorr = nanmean(savePsthCrossCorr,3);
[~,saveInd] = max(meanCorr)

N = (size(meanCorr,1)-1)/2;
lag = (saveInd-N)*binSize;
absLag = abs(lag);

[~,ind] = min(nanstd(absLag));

lag(:,ind)

%%

%N = (size(meanCorr,1)-1)/2;
%lag = (saveInd-N)*binSize;

lag = runningLag*binSize-0.11;
%lag = saveFirstLag*binSize-0.1;

[~,ind] = min(nanstd(absLag));
tInd = 7;

plotN = 141;
delta = 0.8;
tDelta = 0.05;

clf;
subplot(1,2,1);
hold all;

lastMax = 0;
for cInd = 1:plotN
    trace       = nanmean(nanmean(fullPsth(:,cInd,tInd,:),4),2);

    y = trace/(max(trace)+eps) + lastMax;
    plot(binCenters, y)
    lastMax = max(y)-delta;
end

axis([-tDelta 1.6+tDelta 0 lastMax+delta]);

chamod();



subplot(1,2,2);
hold all;

lastMax = 0;
for cInd = 1:plotN
    trace       = nanmean(nanmean(fullPsth(:,cInd,tInd,:),4),2);

    x = binCenters +lag(cInd);
    y = trace/(max(trace)+eps) + lastMax;
    
    plot(x, y)
    lastMax = max(y)-delta;
end


axis([-tDelta 1.6+tDelta 0 lastMax+delta]);
chamod();

%% OK, so we have lags now.

meanCorr = nanmean(savePsthCrossCorr,3);
[~,lagInd] = max(meanCorr);
relLag = max(lagInd) - lagInd;

%lag = (lagInd-N)*binSize;
%absLag = abs(lag);


shiftPsth = nan([size(fullPsth) + [400 0 0 0]]);

for cInd = 1:141
    useInd = [1:size(fullPsth,1)]+relLag(cInd);
    shiftPsth(useInd,cInd,:,:) = fullPsth(:,cInd,:,:);
end

%%

indList;

%meanCorr = nanmean(savePsthCrossCorr,3);
%[~,lagInd] = max(meanCorr);
relLag = indList-min(indList);

%lag = (lagInd-N)*binSize;
%absLag = abs(lag);


shiftPsth = nan([size(fullPsth) + [400 0 0 0]]);

for cInd = 1:141
    useInd = [1:size(fullPsth,1)]+relLag(cInd);
    shiftPsth(useInd,cInd,:,:) = fullPsth(:,cInd,:,:);
end

traceInd = find(~(any(any(any(isnan(shiftPsth),4),3),2)));
usePsth = shiftPsth(traceInd,:,:,:);



tInd = 5;

clf;
im = nanmean(usePsth(:,:,tInd,:),4)';
imagesc(im);

sum(isnan(im(:)))

%%

N = 400;
obsMat = permute(nanmean(shiftPsth(N:end-N, :, :, :),4), [2 3 1]);

matSize = size(obsMat);

[c,s,l] = pca(obsMat(:,:)');


%%
pcInd = 2;

subplot(1,3,1);
imagesc( squeeze(nanmean(obsMat,1)));

subplot(1,3,2);
imagesc( reshape(s(:,pcInd), [matSize(2:3)]) )

subplot(1,3,3);
x = squeeze(nanmean(obsMat,1));
y = reshape(s(:,pcInd), [matSize(2:3)]);
scatter(x(:),y(:));
title(nancorr(x(:),y(:)));

%%

rotTInd = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];

pcInds = [ 1 2];

clf;
hold all;
x = reshape(s(:,pcInds(1)), [matSize(2:3)]);
y = reshape(s(:,pcInds(2)), [matSize(2:3)]);

for index = 0:5:1000
    clf;
    hold all;
    rotInd = [1:100] + index;

    for tInd = rotTInd
        plot(x(tInd,rotInd),y(tInd,rotInd));
    end
    axis square;
    chamod();
    axis([-1 1 -0.7 0.7]);
    drawnow;
    pause(0.1);
end
%title(nancorr(x(:),y(:)));

%%
pcScores = reshape(s(:,:), [matSize(2:3) size(s,2)]);


thisPcCorr = nan(2401,141,141);
for pcInd1 = 1:141
    for pcInd2 = (pcInd1+1):141
        thisCorr = [];
        for tInd = 1:59
            thisCov = xcov(pcScores(tInd,:,pcInd1), pcScores(tInd,:,pcInd2), 'coeff');
            
            thisCorr(:,tInd) = thisCov;
        end
        
        thisPcCorr(:,pcInd1,pcInd2) = nanmean(thisCorr,2);
    end
    pcInd1
end
            
%%

N = (size(thisPcCorr,1)-1)/2;
x = (-N:N)*binSize;

clf;
plot(x, thisPcCorr(:,1,2));



%% Initial latency method

% Find all pairwise
% Find nearest poi






%% Observations

% We see similar frequency content across different PCs for the same
% texture

% Latency differences? 
% Perhaps building up more complex waveforms? 
% Check if the periodicity is consistent across PCs, just phase-shifted?



%% Ideas:

% What if we did the PCA leaving one texture out, and reconstructed the
% expected trace for each neuron?

%%

% NOTE: Check across hemispheres - did the textures flip? Does that change
%   the spatial patterning? Do we need to correct for this or are we good
%   to go?

% Next steps - look at population activity over time. 




%% Cut into a smaller thing, then 

% Cut out 400 ms

% Add X seconds to each trace, then cut out the first ~400 ms

% Perform PCA on the remaining 


%% First pass PCA on the data



%% How far can we push it?


%% Is there a better model for the latent factors?



