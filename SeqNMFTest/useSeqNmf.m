
%%

meanFullPsth = nanmean(fullPsth,4);

size(meanFullPsth)



%% 

% Notes

% First pass, let's try a small range of lags
% Then consider expanding larger lag widths

% First pass, let's try an easy-ish one (city lights)
% Then try expanding to multiple textures

%% 

%% Next steps


% 1) Figure out why I've got this alternating error. Tease that out
% 2) Can we smooth out H?
% 3) 
%%


diffLMat = shiftdim((eye(nLags) + tril(ones(nLags)) - tril(ones(nLags),1)),-2);
diffTMat = eye(nT) + tril(ones(nT)) - tril(ones(nT),1);

%%

tolerance   = 10^-10;
lambda      = 10^-4.5;
lambdaL1W   = 0; %10^-1;
nIterations = 51;

lambdaHSmooth = 0;
lambdaWSmooth = 0;


xMat        = meanFullPsth(:,:,5)';

nNeurons    = size(xMat,1);
trueNT      = size(xMat,2);

nLags       = 250;
nFactors    = 1;


xMat = [zeros(nNeurons,nLags),xMat,zeros(nNeurons,nLags)];
%xMat = [zeros(nNeurons,nLags),xMat];

[~, nT] = size(xMat);


smallNum    = max(xMat(:))*1e-6;


% initialize factors
W = max(xMat(:))*rand(nNeurons, nFactors, nLags); % K factors x N neurons x L lags. We want this to be sparse.
 
% eventually: K factors, T time points x C conditions. No need to be sparse.
% Why do we care about the Frobeneus norm of each row?
H = max(xMat(:))*rand(nFactors,nT)./(sqrt(nT/3)); % normalize so frobenius norm of each row ~ 1

xHat = reconXHat(W, H); 
W = W.*(median(xMat,2)./median(xHat,2));
xHat = reconXHat(W, H); 

% Nice idea for cross-validation: 
% X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit


smoothKernel = ones(1,(2*nLags)-1);  % for factor competition
smallNum = max(xMat(:))*1e-6; 
lastTime = 0;

% Calculate initial cost
iterCost = zeros(nIterations+1, 1);
iterCost(1) = sqrt(mean((xMat(:)-xHat(:)).^2));

lastRun = false;
for iterInd = 1:nIterations
    iterInd
    % Are our updates still changing things? Or have we reached a local
    % minima?
    
    if false && iterInd>5 && (iterCost(iterInd)+tolerance)>nanmean(iterCost((iterInd-5):iterInd-1))
        [iterInd iterCost(iterInd) nanmean(iterCost((iterInd-5):iterInd-1))]
        %cost = cost(1 : iter+1);  % trim vector
        lastRun = true;
        if iterInd>1
            lambda = 0; % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
        end
    end
    
    
    % Normal update terms for convolutional non-negative matrix
    % factorization
    WTX = zeros(nFactors, nT);
    WTXhat = zeros(nFactors, nT);
    for lInd = 1 : nLags
        X_shifted = circshift(xMat,[0,-lInd+1]); 
        xHat_shifted = circshift(xHat,[0,-lInd+1]); 
        WTX = WTX + W(:, :, lInd)' * X_shifted;
        WTXhat = WTXhat + W(:, :, lInd)' * xHat_shifted;
    end   
    
    % Compute regularization terms for H update
    if lambda>0
        dRdH = lambda.*(~eye(nFactors))*conv2(WTX, smoothKernel, 'same');  
    else 
        dRdH = 0; 
    end
    if false && lambdaL1H > 0
        
    else
        lambdaL1H = 0;
    end
    if false && params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH*(~eye(K))*conv2(H, smoothKernel, 'same');
    else
        dHHdH = 0;
    end
    
    if true
        dSmoothdH = exp(lambdaHSmooth*[H(:,1)-H(:,2) (H(:,2:end-1)-(H(:,1:end-2)+H(:,3:end))/2) H(:,end)-H(:,end-1)]);
    else
        dSmoothdH = 0;
    end
    dRdH = dRdH + lambdaL1H + dHHdH; % include L1 sparsity, if specified
    %dRdH = dRdH.*dSmoothdH;
   
    
    
    
    % Update H
    H = H .* WTX ./ (WTXhat + dRdH +eps); % How do we calculate all of these?
        
    norms = sqrt(sum(H.^2, 2))';
    H = H./norms';
    W = W.*norms;
    
    if true % parameter shifting 
        [W, H] = shiftMyFactors(W, H);  
        W = W+smallNum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    


    
    
    %H = diag(1 ./ (norms+eps)) * H;
    %for lInd = 1 : nLags
    %    W(:, :, lInd) = W(:, :, lInd) * diag(norms);
    %end 
    
    
    
    if lambda>0  %  && params.useWupdate
        XS = conv2(xMat, smoothKernel, 'same'); 
    end
    
    if true
        dSmoothdW = exp(lambdaWSmooth.*cat(3,W(:,:,1)-W(:,:,2),(W(:,:,2:end-1)-(W(:,:,1:end-2)+W(:,:,3:end))/2),W(:,:,end)-W(:,:,end-1)));
    else
        dSmoothdW = 0;
    end
        
    
    % Update each W at each lag separately
    for lInd = 1:nLags
        H_shifted = circshift(H,[0,lInd-1]);
        XHT = xMat * H_shifted';
        XhatHT = xHat * H_shifted';
        
        
        if lambda>0 % && params.useWupdate    % Often get similar results with just H update, so option to skip W update
            dRdW = lambda.*XS*(H_shifted')*(~eye(nFactors)); 
        else
            dRdW = 0;
        end
        if ~isempty(lambdaL1W)
            
        else
            lambdaL1W = 0;
        end
        if false && params.lambdaOrthoW>0
            dWWdW = params.lambdaOrthoW*Wflat*(~eye(K));
        else
            dWWdW = 0;
        end

        dRdW = dRdW + lambdaL1W + dWWdW; % include L1 and Worthogonality sparsity, if specified
        %dRdW = dRdW.*dSmoothdW(:,:,lInd);
        
        if(any(dRdW(:) < 0))
            error()
        end
        
        if lInd == 1 
            if iterInd == 5
                %error()
            end
        end
        
        % Update W
        W(:, :, lInd) = W(:, :, lInd) .* XHT ./ (XhatHT + dRdW + eps); % How do we calculate all of these?
        
        
    end
    
    
    % Compute cost
    xHat = reconXHat(W, H); 
    
    
    
    % mask = find(params.M == 0); % find masked (held-out) indices 
    % X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    
    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    if iterCost(iterInd+1) > iterCost(iterInd)
        W2 = W.*(median(xMat,2)./median(xHat,2));
        xHat2 = reconXHat(W2, H); 
        iterCost2 = sqrt(mean((xMat(:)-xHat2(:)).^2));
        
        if iterCost2 < iterCost(iterInd+1)
            W = W2;
            xHat = xHat2;
            iterCost(iterInd+1) = iterCost2;
        end
        
        
    end
    
    
    if lastRun
        break
    end
end

%xMat = xMat(:,nLags+1:end);
%xHat = xHat(:,nLags+1:end);
%H = H(:,nLags+1:end);

xMat = xMat(:,nLags+1:end-nLags);
xHat = xHat(:,nLags+1:end-nLags);
H = H(:,nLags+1:end-nLags);

% Need to measure reconstruction error



%%


factorInd = 1;

clf;
subplot(4,3,1);
plot(iterCost);

subplot(4,3,[4 7 10]);
x = (0:size(W,3))*binSize;
y = 1:size(W,1);
im = squeeze(W(:,factorInd,:));
imagesc(x,y,im);
%caxis([0 nanmean(im(:))*nanstd(im(:))*5]);
caxis([0 quantile(im(:), 0.9)]);


%caxis([-1 1]*nanstd(im(:))*1.2 + nanmean(im(:)));
chamod();

xlabel('time');
ylabel('neuron');
title('neural loading');


subplot(3, 3, [ 2 3])
x = (1:size(H,2))*binSize;
plot(x,H(factorInd,:));
chamod();

xlabel('time');
ylabel('loading');
title('texture loading');


[~,neuronOrder] = sort(nanstd( xMat, [], 2));


subplot(3, 3, [ 5 6])
im = xMat(neuronOrder,:);
imagesc( im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');


subplot(3, 3, [ 8 9])
im = reconXHat(W(:,factorInd,:), H(factorInd,:));
imagesc( im(neuronOrder,:) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');

%% 

export_fig -m1 FirstFactor.png

%%

%%

factorList = 1:2;

[~,neuronOrder] = sort(nanstd( xMat, [], 2));

clf;
subplot(2,1,1);
im = xMat(neuronOrder,:);
imagesc(im);
caxis([0 quantile(im(:), 0.9)]);


xHat = reconXHat(W(:,factorList,:), H(factorList,:)); 

subplot(2,1,2);
im = xHat(neuronOrder,:);
imagesc(im);
caxis([0 quantile(im(:), 0.9)]);

%% New Ideas

% 1) Add in a constant term (or single line term, for adaptation)
%       How do we gradient descend down it?
%       

% 2) Run the algorithm on more than one texture at a time

%% First, add in constant term


tolerance   = 10^-10;
lambda      = 10^-4.5;
lambdaL1W   = 0; %10^-1;
lambdaHSmooth = 0;
lambdaWSmooth = 0;

nIterations = 201;

nFindFactors    = 4;
nFactors        = nFindFactors+1;


rotTInd = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];

useInd = rotTInd;
useInd = 1:59;
%useInd = [1 2 3];

%xMat        = cityLightsTraces';
%xMat        = permute( meanFullPsth(:,:,[5 7]), [2 1 3] );
xMat        = permute( nanmean(fullPsth(:,:,useInd,:),4), [2 1 3] );
%xMat        = permute( meanFullPsth(:,:,[48]), [2 1 3] );

nNeurons    = size(xMat,1);
trueNT      = size(xMat,2);
nConditions = size(xMat,3);

nLags           = 250;



xMat = cat(2,zeros(nNeurons,nLags,nConditions),xMat,zeros(nNeurons,nLags,nConditions));
%xMat = [zeros(nNeurons,nLags),xMat];

[~, nT,~] = size(xMat);


smallNum    = max(xMat(:))*1e-6;


% initialize factors
W = max(xMat(:))*rand(nNeurons,1,1,nFactors,nLags); % K factors x N neurons x L lags. We want this to be sparse.
W(:,1,1,1,:) = repmat(nanmean(W(:,1,1,1,:),3), [1 1 size(W,3)]);

% eventually: K factors, T time points x C conditions. No need to be sparse.
% Why do we care about the Frobeneus norm of each row?
H = rand(1,nT,nConditions,nFactors); % normalize so frobenius norm of each row ~ 1
H(1,:,:,1) = repmat(nanmean(nanmean(H(1,:,:,1),2),3), [1 size(H,2) size(H,3) 1]);

norms = sqrt(sum(H.^2, 2));
H = H./norms;

tic
xHat = reconXHat(W, H); 
t = toc;
[0 t]


W = W.*median(median(xMat,2),3)./median(median(xHat,2),3);

tic
xHat = reconXHat(W, H); 
t = toc;
[0 t]

% Nice idea for cross-validation: 
% X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit


smoothKernel = ones(1,(2*nLags)-1);  % for factor competition
smallNum = max(xMat(:))*1e-6; 
lastTime = 0;

% Calculate initial cost
iterCost = zeros(nIterations+1, 1);
iterCost(1) = sqrt(mean((xMat(:)-xHat(:)).^2));

lastRun = false;
for iterInd = 1:nIterations
    iterInd
    % Are our updates still changing things? Or have we reached a local
    % minima?
    
%     if false && iterInd>5 && (iterCost(iterInd)+tolerance)>nanmean(iterCost((iterInd-5):iterInd-1))
%         [iterInd iterCost(iterInd) nanmean(iterCost((iterInd-5):iterInd-1))]
%         %cost = cost(1 : iter+1);  % trim vector
%         lastRun = true;
%         if iterInd>1
%             lambda = 0; % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
%         end
%     end
    tic
    
    % Normal update terms for convolutional non-negative matrix
    % factorization
    WTX = 0; %zeros(nFactors, nT);
    WTXhat = 0; %zeros(nFactors, nT);
    
    %xMatSum = sum(xMat,3);
    %xHatSum = sum(xHat,3);
    
    
    parfor lInd = 1 : nLags
        X_shifted = circshift(xMat,[0,-lInd+1,0]); 
        xHat_shifted = circshift(xHat,[0,-lInd+1,0]); 
        WTX = WTX + sum(W(:,1,1,:,lInd) .* X_shifted,1);
        WTXhat = WTXhat + sum(W(:,1,1,:,lInd) .* xHat_shifted,1);
    end
    t = toc;
    [iterInd t]
    
    % Compute regularization terms for H update
    if lambda>0
        dRdH = lambda.*permute(sum((shiftdim(~eye(nFactors),-3)).*convn(WTX, smoothKernel, 'same'),4),[1 2 3 5 4]);
    else 
        dRdH = 0; 
    end
    if false && lambdaL1H > 0
        
    else
        lambdaL1H = 0;
    end
    if false && params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH*(~eye(K))*conv2(H, smoothKernel, 'same');
    else
        dHHdH = 0;
    end
    
    if false
        dSmoothdH = exp(lambdaHSmooth*[H(:,1)-H(:,2) (H(:,2:end-1)-(H(:,1:end-2)+H(:,3:end))/2) H(:,end)-H(:,end-1)]);
    else
        dSmoothdH = 0;
    end
    dRdH = dRdH + lambdaL1H + dHHdH; % include L1 sparsity, if specified
    %dRdH = dRdH.*dSmoothdH;
   
    
    
    
    % Update H
    H(1,:,:,1) = H(1,:,:,1) .* nanmean(WTX(1,:,:,1) ./ (WTXhat(1,:,:,1) + dRdH(1,:,:,1) +eps),2); % How do we calculate all of these?
    H(1,:,:,2:end) = H(1,:,:,2:end) .* WTX(1,:,:,2:end) ./ (WTXhat(1,:,:,2:end) + dRdH(1,:,:,2:end) +eps); % How do we calculate all of these?
        
%     norms = sqrt(sum(H.^2, 2))';
%     H = H./norms';
%     W = W.*norms;
    
    if false % parameter shifting 
        [W, H] = shiftMyFactors(W, H);  
        W = W+smallNum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    


    
    
    %H = diag(1 ./ (norms+eps)) * H;
    %for lInd = 1 : nLags
    %    W(:, :, lInd) = W(:, :, lInd) * diag(norms);
    %end 
    
    
    
    if lambda>0  %  && params.useWupdate
        XS = convn(xMat, smoothKernel, 'same'); 
    end
    
    if false
        dSmoothdW = exp(lambdaWSmooth.*cat(3,W(:,:,1)-W(:,:,2),(W(:,:,2:end-1)-(W(:,:,1:end-2)+W(:,:,3:end))/2),W(:,:,end)-W(:,:,end-1)));
    else
        dSmoothdW = 0;
    end
        
    
    % Update each W at each lag separately
    parfor lInd = 1:nLags
        H_shifted = circshift(H,[0,lInd-1,0,0]);
        XHT = sum(sum(xMat .* H_shifted,2),3);
        XhatHT = sum(sum(xHat .* H_shifted,2),3);
        
        
        if lambda>0 % && params.useWupdate    % Often get similar results with just H update, so option to skip W update
            dRdW = lambda.*permute(sum(sum(sum(XS.*H_shifted,2),3).*shiftdim(~eye(nFactors),-3),4), [1 2 3 5 4]); 
        else
            dRdW = 0;
        end
%         if ~isempty(lambdaL1W)
%             
%         else
%             lambdaL1W = 0;
%         end
%         if false %&& params.lambdaOrthoW>0
%             %dWWdW = params.lambdaOrthoW*Wflat*(~eye(K));
%         else
%             dWWdW = 0;
%         end

        %dRdW = dRdW + lambdaL1W + dWWdW; % include L1 and Worthogonality sparsity, if specified
        %dRdW = dRdW.*dSmoothdW(:,:,lInd);
        
        if(any(dRdW(:) < 0))
            error()
        end
        
        if lInd == 1 
            if iterInd == 5
                %error()
            end
        end
        
        % Update W
        W(:,1,1,:,lInd) = W(:,1,1,:,lInd) .* XHT ./ (XhatHT + dRdW + eps); % How do we calculate all of these?
    end
    t = toc;
    [iterInd t]
    W(:,1,1,1,:) = repmat(nanmean(W(:,1,1,1,:),5), [1 1 1 1 size(W,5)]);
    
    
    % Compute cost
    xHat = reconXHat(W, H); 
    
    t = toc;
    [iterInd t]
    
    % mask = find(params.M == 0); % find masked (held-out) indices 
    % X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    
    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    if iterCost(iterInd+1) > iterCost(iterInd)
        W2 = W.*median(median(xMat,2),3)./median(median(xHat,2),3);
        xHat2 = reconXHat(W2, H); 
        iterCost2 = sqrt(mean((xMat(:)-xHat2(:)).^2));
        
        if iterCost2 < iterCost(iterInd+1)
            W = W2;
            xHat = xHat2;
            iterCost(iterInd+1) = iterCost2;
            
            clear xHat2;
            clear W2;
        end
        
        
    end
    t = toc;
    [iterInd t]
    
    if lastRun
        break
    end
end

clear X_shifted;
clear xHat_shifted;
clear XS;
clear dRdH;
clear dSmoothdH;
clear WTX;
clear WTXhat;

%xMat = xMat(:,nLags+1:end);
%xHat = xHat(:,nLags+1:end);
%H = H(:,nLags+1:end);

xMat = xMat(:,nLags+1:end-nLags,:);
xHat = xHat(:,nLags+1:end-nLags,:);
H = H(:,nLags+1:end-nLags,:,:);

% Need to measure reconstruction error


%%

save('seqNMFVar', 'xMat', 'xHat', 'W', 'H');

%%
load('seqNMFVar', 'xMat', 'xHat', 'W', 'H');

wFull = W;
hFull = H;
%%

clf;

x = W(:,1,1,3,:);
y = W(:,1,1,5,:);

scatter(squeeze(nanmean(x,1)), squeeze(nanmean(y,1)));

%%

factorInd = 2;
tInd = 4;



[~,neuronOrder] = sort(nanmean(nanstd( xMat, [], 2),3));



clf;
subplot(4,3,1);
plot(iterCost);

axis([0 length(iterCost)+1 0 max(iterCost)*1.05]);

subplot(4,3,[4 7 10]);
x = (0:size(W,5))*binSize;
y = 1:size(W,1);
im = squeeze(W(neuronOrder,1,1,factorInd,:));
imagesc(x,y,im);
%caxis([0 nanmean(im(:))*nanstd(im(:))*5]);
caxis([0 quantile(im(:), 0.9)]);


%caxis([-1 1]*nanstd(im(:))*1.2 + nanmean(im(:)));
chamod();

xlabel('time');
ylabel('neuron');
title('neural loading');


subplot(3, 3, [ 2 3])
x = (1:size(H,2))*binSize;
plot(x,squeeze(H(1,:,tInd,factorInd)));
chamod();

xlabel('time');
ylabel('loading');
%title('texture loading');
title( cdaData.fullRun.textures{tInd});



subplot(3, 3, [ 5 6])
im = xMat(neuronOrder,:,tInd);
imagesc( im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');


subplot(3, 3, [ 8 9])
im = reconXHat(W(:,1,1,factorInd,:), H(1,:,tInd,factorInd));
imagesc( im(neuronOrder,:) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');


%%



factorInd = 1:5;
tInd = 35;

[~,neuronOrder] = sort(nanmean(nanstd( xMat, [], 2),3));

clf;
subplot(3,1,1);
hold all;
for i=1:5
    plot(H(1,:,tInd,i));
end
title(cdaData.fullRun.textures{tInd});



subplot(3,1,2)
im = xMat(neuronOrder,:,tInd);
imagesc( im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');


subplot(3,1,3)
im = reconXHat(W(:,1,1,factorInd,:), H(1,:,tInd,factorInd));
imagesc( im(neuronOrder,:) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');

%%

factorInd = 2;
clf;
hold all;
for cInd = (1:10)+130
    plot( squeeze(W(cInd,1,1,factorInd,:)));
end



%% New ideas

% 1) One factor at a time. (Plus the constant)
% 2) Take off the seqNMF regularizer (it's for birdsong)
% 3) Prime the edges of W. Try to get it to converge to being centered at 0.
% 4) Smoothness! Penalize the square of the second derivative

%% New new issues

% 1) The edges of W keep riding up. 
% 2) The left edge of the predicted data and the right edge of H are set to
%       zero. Some systematic issue here.
% 3) The constant element keeps falling to 0. Can we recover this with
%       standard NMF, do some calculation on the time-averaged data?


%%


tolerance       = 10^-5;
%lambda          = 10^-4.5;
lambda          = 0;
lambdaL1W       = 10^-1;
lambdaHSmooth   = 10.^1.6;
lambdaWSmooth   = 10.^1.6;



nIterations     = 251;

nFindFactors    = 1;
nFactors        = nFindFactors+1;


rotTInd = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];
%useTList = rotTInd;
%useTList = 1:10;
useTList = 1:59;
%useTList = [4 5 7 41];
%useTList = randsample(59,10);


%useInd = [1 2 3];

%xMat        = cityLightsTraces';
%xMat        = permute( meanFullPsth(:,:,[5 7]), [2 1 3] );
xMat        = permute( nanmean(fullPsth(:,:,useTList,:),4), [2 1 3] );
%xMat        = permute( meanFullPsth(:,:,[48]), [2 1 3] );

nNeurons    = size(xMat,1);
trueNT      = size(xMat,2);
nConditions = size(xMat,3);

nLags           = 250;


xMean = ones(nNeurons,nLags,nConditions).*nanmean(xMat,2);
xMat = cat(2,xMean,xMat,xMean);
%xMat = [zeros(nNeurons,nLags),xMat];

[~, nT,~] = size(xMat);


smallNum    = max(xMat(:))*1e-6;


% initialize factors
W = max(xMat(:))*rand(nNeurons,1,1,nFactors,nLags); % K factors x N neurons x L lags. 
midLag = floor(size(W,5)/2);
W(:,1,1,2:end, 1:midLag) = W(:,1,1,2:end, 1:midLag) .* shiftdim(linspace(0.5,1,midLag), -3);
W(:,1,1,2:end, midLag+1:end) = W(:,1,1,2:end, midLag+1:end) .* shiftdim(linspace(1,0.5,midLag), -3);
W(:,1,1,1,:) = repmat(nanmean(W(:,1,1,1,:),5), [1 1 1 1 size(W,5)]);

W(:,1,1,:,[1 end]) = 0;


% eventually: K factors, T time points x C conditions. No need to be sparse.
% Why do we care about the Frobeneus norm of each row?
H = rand(1,nT,nConditions,nFactors); % normalize so frobenius norm of each row ~ 1
H(1,:,:,2:end) = H(1,:,:,2:end).*nanmean(xMat,1);
H(1,:,:,1) = repmat(nanmean(nanmean(H(1,:,:,1),2),3), [1 size(H,2) size(H,3) 1]);


norms = sqrt(sum(H.^2, 2));
H = H./norms;

tic
xHat = reconXHat(W, H); 
t = toc;
[0 t]


W = W.*median(median(xMat,2),3)./median(median(xHat,2),3);

tic
xHat = reconXHat(W, H); 
t = toc;
[0 t]

% Nice idea for cross-validation: 
% X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit


smoothKernel = ones(1,(2*nLags)-1);  % for factor competition
smallNum = max(xMat(:))*1e-6; 
lastTime = 0;

% Calculate initial cost
iterCost = zeros(nIterations+1, 1);
iterCost(1) = sqrt(mean((xMat(:)-xHat(:)).^2));

lastRun = false;
for iterInd = 1:nIterations
    iterInd
    % Are our updates still changing things? Or have we reached a local
    % minima?
    
%     if false && iterInd>5 && (iterCost(iterInd)+tolerance)>nanmean(iterCost((iterInd-5):iterInd-1))
%         [iterInd iterCost(iterInd) nanmean(iterCost((iterInd-5):iterInd-1))]
%         %cost = cost(1 : iter+1);  % trim vector
%         lastRun = true;
%         if iterInd>1
%             lambda = 0; % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
%         end
%     end
    tic
    
    % Normal update terms for convolutional non-negative matrix
    % factorization
    WTX = 0; %zeros(nFactors, nT);
    WTXhat = 0; %zeros(nFactors, nT);
    
    %xMatSum = sum(xMat,3);
    %xHatSum = sum(xHat,3);
    
    
    parfor lInd = 1 : nLags
        X_shifted = circshift(xMat,[0,-lInd+1,0]); 
        xHat_shifted = circshift(xHat,[0,-lInd+1,0]); 
        WTX = WTX + sum(W(:,1,1,:,lInd) .* X_shifted,1);
        WTXhat = WTXhat + sum(W(:,1,1,:,lInd) .* xHat_shifted,1);
    end
    t = toc;
    [iterInd t]
    
    % Compute regularization terms for H update
    if lambda>0
        dRdH = lambda.*permute(sum((shiftdim(~eye(nFactors),-3)).*convn(WTX, smoothKernel, 'same'),4),[1 2 3 5 4]);
    else 
        dRdH = zeros(1,1,1,size(H,4)); 
    end
    if false && lambdaL1H > 0
        
    else
        lambdaL1H = 0;
    end
    if false && params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH*(~eye(K))*conv2(H, smoothKernel, 'same');
    else
        dHHdH = 0;
    end
    
    if true
        %dSmoothdH = exp(lambdaHSmooth*[H(:,1)-H(:,2) (H(:,2:end-1)-(H(:,1:end-2)+H(:,3:end))/2) H(:,end)-H(:,end-1)]);
        %dSmoothdH = 1;
                
        dSmoothdH = lambdaHSmooth*cat(2, 2*H(:,1,:,:) - 3*H(:,1,:,:) + H(:,3,:,:), ...
            -3*H(:,1,:,:)+6*H(:,2,:,:)-4*H(:,3,:,:)+H(:,4,:,:), ...
            H(:,1:end-4,:,:) - 4*H(:,2:end-3,:,:) + 6*H(:,3:end-2,:,:) - 4*H(:,4:end-1,:,:) + H(:,5:end,:,:), ...
            H(:,1,:,:)-4*H(:,2,:,:)+6*H(:,end-1,:,:)-3*H(:,end,:,:), ...
            H(:,end-2,:,:) - 3*H(:,end-1,:,:) + 2*H(:,end,:,:));
        
        dSmoothdH(dSmoothdH < 0) = 0;
            
    else
        dSmoothdH = 0;
    end
    %dRdH = dRdH + lambdaL1H + dHHdH; % include L1 sparsity, if specified
    dRdH = dRdH + dSmoothdH;
   
    
                
    if(any(dSmoothdH(:) < 0))
        error()
    end

    
    % Update H
    denom = (WTXhat + dRdH +eps);
    if(any(denom(:) < 0))
        error()
    end    
    
    H(1,:,:,1) = H(1,:,:,1) .* nanmean(WTX(1,:,:,1) ./ denom(1,:,:,1),2); % How do we calculate all of these?
    H(1,:,:,2:end) = H(1,:,:,2:end) .* WTX(1,:,:,2:end) ./ denom(1,:,:,2:end); % How do we calculate all of these?
        
%     norms = sqrt(sum(H.^2, 2))';
%     H = H./norms';
%     W = W.*norms;
    
    if false % parameter shifting 
        [W, H] = shiftMyFactors(W, H);  
        W = W+smallNum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    


    
    
    %H = diag(1 ./ (norms+eps)) * H;
    %for lInd = 1 : nLags
    %    W(:, :, lInd) = W(:, :, lInd) * diag(norms);
    %end 
    
    
    
    if lambda>0  %  && params.useWupdate
        XS = convn(xMat, smoothKernel, 'same'); 
    end
    
    if true
        %dSmoothdW = exp(lambdaWSmooth.*cat(3,W(:,:,1)-W(:,:,2),(W(:,:,2:end-1)-(W(:,:,1:end-2)+W(:,:,3:end))/2),W(:,:,end)-W(:,:,end-1)));
        
        dSmoothdW = lambdaWSmooth*cat(5, 2*W(:,:,:,:,1) - 3*W(:,:,:,:,1) + W(:,:,:,:,3), ...
            -3*W(:,:,:,:,1)+6*W(:,:,:,:,2)-4*W(:,:,:,:,3)+W(:,:,:,:,4), ...
            W(:,:,:,:,1:end-4) - 4*W(:,:,:,:,2:end-3) + 6*W(:,:,:,:,3:end-2) - 4*W(:,:,:,:,4:end-1) + W(:,:,:,:,5:end), ...
            W(:,:,:,:,1)-4*W(:,:,:,:,2)+6*W(:,:,:,:,end-1)-3*W(:,:,:,:,end), ...
            W(:,:,:,:,end-2) - 3*W(:,:,:,:,end-1) + 2*W(:,:,:,:,end));
            
                       
        dSmoothdW(dSmoothdW < 0) = 0;
        
    else
        dSmoothdW = 0;
    end
        
    
    % Update each W at each lag separately
    parfor lInd = 1:nLags
        H_shifted = circshift(H,[0,lInd-1,0,0]);
        XHT = sum(sum(xMat .* H_shifted,2),3);
        XhatHT = sum(sum(xHat .* H_shifted,2),3);
        
        
        if lambda>0 % && params.useWupdate    % Often get similar results with just H update, so option to skip W update
            dRdW = lambda.*permute(sum(sum(sum(XS.*H_shifted,2),3).*shiftdim(~eye(nFactors),-3),4), [1 2 3 5 4]); 
        else
            dRdW = 0;
        end
%         if ~isempty(lambdaL1W)
%             
%         else
%             lambdaL1W = 0;
%         end
%         if false %&& params.lambdaOrthoW>0
%             %dWWdW = params.lambdaOrthoW*Wflat*(~eye(K));
%         else
%             dWWdW = 0;
%         end

        %dRdW = dRdW + lambdaL1W + dWWdW; % include L1 and Worthogonality sparsity, if specified
        %dRdW = dRdW.*dSmoothdW(:,:,lInd);

        
        if lInd == 1 
            if iterInd == 5
                %error()
            end
        end
        
        % Update W
        %W(:,1,1,:,lInd) = W(:,1,1,:,lInd) .* XHT ./ (XhatHT + dRdW + eps); % How do we calculate all of these?
        denom = (XhatHT + dRdW + lambdaL1W + dSmoothdW(:,:,:,:,lInd) + eps);
        if(any(denom(:) < 0))
            error()
        end   
        W(:,1,1,:,lInd) = W(:,1,1,:,lInd) .* XHT ./ denom; 
        
    end
    t = toc;
    [iterInd t]
    W(:,1,1,1,:) = repmat(nanmean(W(:,1,1,1,:),5), [1 1 1 1 size(W,5)]);
    
    
    % Compute cost
    xHat = reconXHat(W, H); 
    
    t = toc;
    [iterInd t]
    
    % mask = find(params.M == 0); % find masked (held-out) indices 
    % X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    
    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    if iterCost(iterInd+1) > iterCost(iterInd)
        W2 = W.*median(median(xMat,2),3)./median(median(xHat,2),3);
        xHat2 = reconXHat(W2, H); 
        iterCost2 = sqrt(mean((xMat(:)-xHat2(:)).^2));
        
        if iterCost2 < iterCost(iterInd+1)
            W = W2;
            xHat = xHat2;
            iterCost(iterInd+1) = iterCost2;
            
            clear xHat2;
            clear W2;
        end
        
        
    end
    t = toc;
    [iterInd t]
    
    if lastRun
        break
    end
end

clear X_shifted;
clear xHat_shifted;
clear XS;
clear dRdH;
clear dSmoothdH;
clear WTX;
clear WTXhat;
clear xHat2;

%xMat = xMat(:,nLags+1:end);
%xHat = xHat(:,nLags+1:end);
%H = H(:,nLags+1:end);

xMat = xMat(:,nLags+1:end-nLags,:);
xHat = xHat(:,nLags+1:end-nLags,:);
%H = H(:,nLags+1:end-nLags,:,:);

% Need to measure reconstruction error

%%

save('seqNMFVar', 'xMat', 'xHat', 'W', 'H');

%%

load('seqNMFVar', 'xMat', 'xHat', 'W', 'H');

useTList = 1:59;

%%

factorInd = 2;
tInd = 14;



[~,neuronOrder] = sort(nanmean(nanstd( xMat, [], 2),3));



clf;
subplot(6,3,1);
plot(iterCost);
axis([0 length(iterCost)+1 0 max(iterCost)*1.05]);


x = (1:size(W,5))*binSize;

subplot(6,3,[4 7]);
vals = squeeze(W(neuronOrder((end-3):end),1,1,factorInd,:))';
plot(x,vals);
axis([0 size(W,5)*binSize 0 max(vals(:)+eps)*1.05]);
chamod();

subplot(6,3,[10 13 16]);
y = 1:size(W,1);
im = squeeze(W(neuronOrder,1,1,factorInd,:));
imagesc([0 x],y,im);
%caxis([0 nanmean(im(:))*nanstd(im(:))*5]);
caxis([0 quantile(im(:), 0.9)]);


%caxis([-1 1]*nanstd(im(:))*1.2 + nanmean(im(:)));
chamod();

xlabel('time');
ylabel('neuron');
title('neural loading');


subplot(3, 3, [ 2 3])
x = (1:size(H(1,nLags+1:end-nLags,tInd,factorInd),2))*binSize;
plot(x,squeeze(H(1,nLags+1:end-nLags,tInd,factorInd)));
chamod();

xlabel('time');
ylabel('loading');
%title('texture loading');
title( cdaData.fullRun.textures{useTList(tInd)});



subplot(3, 3, [ 5 6])
im = xMat(neuronOrder,:,tInd);
imagesc(x,1:141,im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');
chamod();


subplot(3, 3, [ 8 9])
im = reconXHat(W(:,1,1,factorInd,:), H(1,:,tInd,factorInd));
imagesc(x,1:141,im(neuronOrder,nLags+1:end-nLags) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');
chamod();

%% Next steps:

% For each cell and texture, convolve the texture (H) with the data (xMat)
% Get the output kernel
% Compare to the measured kernel
% Find a best fit kernel
xMat        = permute( nanmean(fullPsth(:,:,useTList,:),4), [2 1 3] );

thisCoeff = [];
for cInd = 1:141
    for tInd = 1:59
        thisConv = xcov(H(1,:,tInd,2), xMat(cInd,:,tInd), 'none');
        thisCoeff(cInd,:,tInd) = thisConv;
    end
end


%%

cInd = 29;
tInd = 5;

tMid = 0.075;

clf;
subplot(2,1,1);
hold all;
x = (1:size(xMat,2))*binSize;
plot(x,xMat(cInd,:,tInd));

x = ((1:size(H,2))-nLags)*binSize;
plot(x,H(1,:,tInd,2));
title(cdaData.fullRun.textures{tInd});


corrVal = nanmean(nanmean(thisCoeff(cInd,:,:),3),1);

subplot(2,1,2);
hold all;
x = ((1:size(thisCoeff,2))-size(thisCoeff,2)/2)*binSize;
plot(x, corrVal);

useInd = find(x<0.25 & x > -0.25);
[~,maxInd] = max(nanmean(nanmean(thisCoeff(cInd,useInd,:),3),1)./(0.01+(x(useInd)-tMid).^2));
maxVal = corrVal(useInd(maxInd));
tVal = x(useInd(maxInd));

sigma = 0.01;
yGauss = gaussFunc(x,tVal,sigma)*maxVal;
plot(x, yGauss);

fun = @(x,xdata)( x(1)*gaussFunc(xdata,x(2),x(3)));

p0 = [maxVal, tVal, sigma];
p = lsqcurvefit(fun,p0,x,corrVal);

plot(x, fun(p,x));


%(1./sigma).*(((x-tVal)./sigma).^2);


%plot(x,(corrVal-yGauss).*(1./sigma).*(((x-tVal)./sigma).^2).*yGauss/10)

%1./sum((corrVal-yGauss).*(1./sigma).*(((x-tVal)./sigma).^2).*yGauss)


title(max(thisCoeff(cInd,:,tInd)) ./ sqrt(nanmean(thisCoeff(cInd,:,tInd).^2)));
xlim([-250 250]*binSize + tMid);


chamod();
box off;

%%

x = ((1:size(thisCoeff,2))-size(thisCoeff,2)/2)*binSize;
fun = @(x,xdata)( x(1)*gaussFunc(xdata,x(2),x(3)));
sigma = 0.01;

saveGaussParam = [];
for cInd = 1:141
    corrVal = nanmean(nanmean(thisCoeff(cInd,:,:),3),1);


    useInd = find(x<0.25 & x > -0.25);
    [~,maxInd] = max(nanmean(nanmean(thisCoeff(cInd,useInd,:),3),1)./(0.01+(x(useInd)-tMid).^2));
    maxVal = corrVal(useInd(maxInd));
    tVal = x(useInd(maxInd));

    
    p0 = [maxVal, tVal, sigma];
    p = lsqcurvefit(fun,p0,x,corrVal);

    saveGaussParam(:,cInd) = p;
end



%%


clf;
hold all;
plot(squeeze(W(neuronOrder(end),1,1,2,:)))
plot(squeeze(dSmoothdW(neuronOrder(end),1,1,2,:))/lambdaWSmooth)

%%


saveR = [];
saveErr = [];
for cInd = 1:141
    for tInd = 1:5
        x = xMat(cInd,:,tInd);
        y = xHat(cInd,:,tInd);
        saveR(cInd,tInd) = nancorr(x,y);
        saveErr(cInd,tInd) = sum( (x(:)-y(:)).^2 ) / sum( (x(:) - nanmean(x(:))).^2 );
    end
end

%%
tInd = 1;
cInd = 1:141;
cInd = 49;

x = xMat(neuronOrder(cInd),:,tInd);
xHat = reconXHat(W(:,1,1,factorInd,:), H(1,:,tInd,factorInd));
y = xHat(neuronOrder(cInd),nLags+1:end-nLags);
clf;
hold all;

sh = plot(x(:), y(:));
set(sh, 'Marker', '.');

line([0 0.1], [0 0.1]);

%set(gca, 'XScale', 'log', 'YScale','log');

%%


clf;
hold all;

cInd = 101;
x = (-(nLags-1)/2:(nLags-1)/2)*binSize;
y = squeeze(W(neuronOrder(cInd),1,1,2,:))';

com = sum(x.*y)/sum(y);
comVar = sqrt(sum( y.*(x-com).^2)/sum(y));



plot(x,y)

yFit = gaussFunc(x,com, comVar);
yFit = yFit.*sum(y)/sum(yFit);
plot(x,yFit);

%%
% 
% x = (-(nLags-1)/2:(nLags-1)/2)*binSize;
% 
% saveGaussParam = [];
% for cInd = 1:141
%     y = squeeze(W(cInd,1,1,2,:))';
% 
%     com = sum(x.*y)/sum(y);
%     comVar = sqrt(sum( y.*(x-com).^2)/sum(y));
% 
% 
%     amp = sum(y)/sum(yFit);
%     
%     saveGaussParam(:,cInd) = [com comVar amp];
% end
% 

%% OK now we have Gaussians

% We're trying to predict X from H and W
% Convolve H with W

% Max correlation lag?









%% Fit everything, but prime the Ws as Gaussian and the Hs as what we got before



tolerance       = 10^-5;
%lambda          = 10^-4.5;
lambda          = 0;
lambdaL1W       = 10^-1;
lambdaHSmooth   = 10.^1.6;
lambdaWSmooth   = 10.^1.6;



nIterations     = 51;

nFindFactors    = 1;
nFactors        = 1;


rotTInd = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];
%useTList = rotTInd;
%useTList = 1:10;
%useTList = 1:59;
useTList = [4 5 7 41];
%useTList = randsample(59,10);


%useInd = [1 2 3];

%xMat        = cityLightsTraces';
%xMat        = permute( meanFullPsth(:,:,[5 7]), [2 1 3] );
xMat        = permute( nanmean(fullPsth(:,:,useTList,:),4), [2 1 3] );
%xMat        = permute( meanFullPsth(:,:,[48]), [2 1 3] );

nNeurons    = size(xMat,1);
trueNT      = size(xMat,2);
nConditions = size(xMat,3);

nLags           = 350;


xMean = ones(nNeurons,nLags,nConditions).*nanmean(xMat,2);
xMat = cat(2,xMean,xMat,xMean);
%xMat = [zeros(nNeurons,nLags),xMat];

[~, nT,~] = size(xMat);


smallNum    = max(xMat(:))*1e-6;

%

shiftTime = median(saveGaussParam(2,:));
nShiftBins = round(shiftTime/binSize);

saveGaussParam;
x = ((1:nLags)-nLags/2)*binSize;
fun = @(x,xdata)( x(1)*gaussFunc(xdata,x(2),x(3)));
sigma = 0.01;
WNew = [];
for cInd = 1:141
    WNew(cInd,1,1,1,:) = fun(saveGaussParam(:,cInd).*[1 1 0.5]'-[0 shiftTime 0]',x);
end

% initialize factors
% WNew = max(xMat(:))*rand(nNeurons,1,1,nFactors,nLags); % K factors x N neurons x L lags. 
% midLag = floor(size(WNew,5)/2);
% WNew(:,1,1,:, 1:midLag) = WNew(:,1,1,:, 1:midLag) .* shiftdim(linspace(0.5,1,midLag), -3);
% WNew(:,1,1,:, midLag+1:end) = WNew(:,1,1,:, midLag+1:end) .* shiftdim(linspace(1,0.5,midLag), -3);


% eventually: K factors, T time points x C conditions. No need to be sparse.
% Why do we care about the Frobeneus norm of each row?

%HNew = rand(1,nT,nConditions,nFactors); % normalize so frobenius norm of each row ~ 1
%HNew(1,:,:,2:end) = HNew(1,:,:,2:end).*nanmean(xMat,1);
%HNew(1,:,:,1) = repmat(nanmean(nanmean(HNew(1,:,:,1),2),3), [1 size(HNew,2) size(HNew,3) 1]);


HNew = zeros(1,nT,nConditions,nFactors); % normalize so frobenius norm of each row ~ 1


newShift = round((size(HNew,2)-size(H,2))/2 + shiftTime/binSize);



randInd = 1:size(HNew,2);
hCopyInd = (1:size(H,2))+newShift-1;
randInd(hCopyInd) = [];


HNew(1,hCopyInd,:,1) = H(1,:,useTList,2);
HNew(1,randInd,:,:) = rand(1,length(randInd),size(HNew,3), size(HNew,4)).*(nanmean(H(1,:,useTList,2),2)+nanstd(H(1,:,useTList,2),[],2));





norms = sqrt(sum(HNew.^2, 2));
HNew = HNew./norms;

tic
xHat = reconXHat(WNew, HNew); 
t = toc;
[0 t]


WNew = WNew.*median(median(xMat,2),3)./median(median(xHat,2),3);

tic
xHat = reconXHat(WNew, HNew); 
t = toc;
[0 t]

% Nice idea for cross-validation: 
% X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit


smoothKernel = ones(1,(2*nLags)-1);  % for factor competition
smallNum = max(xMat(:))*1e-6; 
lastTime = 0;

% Calculate initial cost
iterCost = zeros(nIterations+1, 1);
iterCost(1) = sqrt(mean((xMat(:)-xHat(:)).^2));

lastRun = false;
for iterInd = 1:nIterations
    iterInd
    % Are our updates still changing things? Or have we reached a local
    % minima?
    
%     if false && iterInd>5 && (iterCost(iterInd)+tolerance)>nanmean(iterCost((iterInd-5):iterInd-1))
%         [iterInd iterCost(iterInd) nanmean(iterCost((iterInd-5):iterInd-1))]
%         %cost = cost(1 : iter+1);  % trim vector
%         lastRun = true;
%         if iterInd>1
%             lambda = 0; % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
%         end
%     end
    tic
    
    % Normal update terms for convolutional non-negative matrix
    % factorization
    WTX = 0; %zeros(nFactors, nT);
    WTXhat = 0; %zeros(nFactors, nT);
    
    %xMatSum = sum(xMat,3);
    %xHatSum = sum(xHat,3);
    
    
    parfor lInd = 1 : nLags
        X_shifted = circshift(xMat,[0,-lInd+1,0]); 
        xHat_shifted = circshift(xHat,[0,-lInd+1,0]); 
        WTX = WTX + sum(WNew(:,1,1,:,lInd) .* X_shifted,1);
        WTXhat = WTXhat + sum(WNew(:,1,1,:,lInd) .* xHat_shifted,1);
    end
    t = toc;
    [iterInd t]
    
    % Compute regularization terms for H update
    if lambda>0
        dRdH = lambda.*permute(sum((shiftdim(~eye(nFactors),-3)).*convn(WTX, smoothKernel, 'same'),4),[1 2 3 5 4]);
    else 
        dRdH = zeros(1,1,1,size(HNew,4)); 
    end
    if false && lambdaL1H > 0
        
    else
        lambdaL1H = 0;
    end
    if false && params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH*(~eye(K))*conv2(HNew, smoothKernel, 'same');
    else
        dHHdH = 0;
    end
    
    if true
        %dSmoothdH = exp(lambdaHSmooth*[H(:,1)-H(:,2) (H(:,2:end-1)-(H(:,1:end-2)+H(:,3:end))/2) H(:,end)-H(:,end-1)]);
        %dSmoothdH = 1;
                
        dSmoothdH = lambdaHSmooth*cat(2, 2*HNew(:,1,:,:) - 3*HNew(:,1,:,:) + HNew(:,3,:,:), ...
            -3*HNew(:,1,:,:)+6*HNew(:,2,:,:)-4*HNew(:,3,:,:)+HNew(:,4,:,:), ...
            HNew(:,1:end-4,:,:) - 4*HNew(:,2:end-3,:,:) + 6*HNew(:,3:end-2,:,:) - 4*HNew(:,4:end-1,:,:) + HNew(:,5:end,:,:), ...
            HNew(:,1,:,:)-4*HNew(:,2,:,:)+6*HNew(:,end-1,:,:)-3*HNew(:,end,:,:), ...
            HNew(:,end-2,:,:) - 3*HNew(:,end-1,:,:) + 2*HNew(:,end,:,:));
        
        dSmoothdH(dSmoothdH < 0) = 0;
            
    else
        dSmoothdH = 0;
    end
    %dRdH = dRdH + lambdaL1H + dHHdH; % include L1 sparsity, if specified
    dRdH = dRdH + dSmoothdH;
   
    
                
    if(any(dSmoothdH(:) < 0))
        error()
    end

    
    % Update H
    denom = (WTXhat + dRdH +eps);
    if(any(denom(:) < 0))
        error()
    end    
    
    %H(1,:,:,1) = H(1,:,:,1) .* nanmean(WTX(1,:,:,1) ./ denom(1,:,:,1),2); % How do we calculate all of these?
    HNew(1,:,:,:) = HNew(1,:,:,:) .* WTX(1,:,:,:) ./ denom(1,:,:,:); % How do we calculate all of these?
        
%     norms = sqrt(sum(H.^2, 2))';
%     H = H./norms';
%     W = W.*norms;
    
    if false % parameter shifting 
        [WNew, HNew] = shiftMyFactors(W, HNew);  
        WNew = WNew+smallNum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    


    
    
    %H = diag(1 ./ (norms+eps)) * H;
    %for lInd = 1 : nLags
    %    W(:, :, lInd) = W(:, :, lInd) * diag(norms);
    %end 
    
    
    
    if lambda>0  %  && params.useWupdate
        XS = convn(xMat, smoothKernel, 'same'); 
    end
    
    if true
        %dSmoothdW = exp(lambdaWSmooth.*cat(3,W(:,:,1)-W(:,:,2),(W(:,:,2:end-1)-(W(:,:,1:end-2)+W(:,:,3:end))/2),W(:,:,end)-W(:,:,end-1)));
        
        dSmoothdW = lambdaWSmooth*cat(5, 2*WNew(:,:,:,:,1) - 3*WNew(:,:,:,:,1) + WNew(:,:,:,:,3), ...
            -3*WNew(:,:,:,:,1)+6*WNew(:,:,:,:,2)-4*WNew(:,:,:,:,3)+WNew(:,:,:,:,4), ...
            WNew(:,:,:,:,1:end-4) - 4*WNew(:,:,:,:,2:end-3) + 6*WNew(:,:,:,:,3:end-2) - 4*WNew(:,:,:,:,4:end-1) + WNew(:,:,:,:,5:end), ...
            WNew(:,:,:,:,1)-4*WNew(:,:,:,:,2)+6*WNew(:,:,:,:,end-1)-3*WNew(:,:,:,:,end), ...
            WNew(:,:,:,:,end-2) - 3*WNew(:,:,:,:,end-1) + 2*WNew(:,:,:,:,end));
                       
        dSmoothdW(dSmoothdW < 0) = 0;
        
    else
        dSmoothdW = 0;
    end
        
    
    % Update each W at each lag separately
    parfor lInd = 1:nLags
        H_shifted = circshift(HNew,[0,lInd-1,0,0]);
        XHT = sum(sum(xMat .* H_shifted,2),3);
        XhatHT = sum(sum(xHat .* H_shifted,2),3);
        
        
        if lambda>0 % && params.useWupdate    % Often get similar results with just H update, so option to skip W update
            dRdW = lambda.*permute(sum(sum(sum(XS.*H_shifted,2),3).*shiftdim(~eye(nFactors),-3),4), [1 2 3 5 4]); 
        else
            dRdW = 0;
        end


        
        if lInd == 1 
            if iterInd == 5
                %error()
            end
        end
        
        % Update W
        %W(:,1,1,:,lInd) = W(:,1,1,:,lInd) .* XHT ./ (XhatHT + dRdW + eps); % How do we calculate all of these?
        denom = (XhatHT + dRdW + lambdaL1W + dSmoothdW(:,:,:,:,lInd) + eps);
        if(any(denom(:) < 0))
            error()
        end   
        WNew(:,1,1,:,lInd) = WNew(:,1,1,:,lInd) .* XHT ./ denom; 
        
    end
    t = toc;
    [iterInd t]
    %WNew(:,1,1,1,:) = repmat(nanmean(WNew(:,1,1,1,:),5), [1 1 1 1 size(WNew,5)]);
    
    
    % Compute cost
    xHat = reconXHat(WNew, HNew); 
    
    t = toc;
    [iterInd t]
    
    % mask = find(params.M == 0); % find masked (held-out) indices 
    % X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    
    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    if iterCost(iterInd+1) > iterCost(iterInd)
        W2 = WNew.*median(median(xMat,2),3)./median(median(xHat,2),3);
        xHat2 = reconXHat(W2, HNew); 
        iterCost2 = sqrt(mean((xMat(:)-xHat2(:)).^2));
        
        if iterCost2 < iterCost(iterInd+1)
            WNew = W2;
            xHat = xHat2;
            iterCost(iterInd+1) = iterCost2;
            
            clear xHat2;
            clear W2;
        end
        
        
    end
    t = toc;
    [iterInd t]
    
    
    
    
    

clf;
subplot(6,3,1);
plot(iterCost);
axis([0 length(iterCost)+1 0 max(iterCost)*1.05]);


x = (1:size(WNew,5))*binSize;

subplot(6,3,[4 7]);
vals = squeeze(WNew(neuronOrder((end-3):end),1,1,factorInd,:))';
plot(x,vals);
axis([0 size(WNew,5)*binSize 0 max(vals(:)+eps)*1.05]);
chamod();

subplot(6,3,[10 13 16]);
y = 1:size(WNew,1);
im = squeeze(WNew(neuronOrder,1,1,factorInd,:));
imagesc([0 x],y,im);
%caxis([0 nanmean(im(:))*nanstd(im(:))*5]);
caxis([0 quantile(im(:), 0.9)]);


%caxis([-1 1]*nanstd(im(:))*1.2 + nanmean(im(:)));
chamod();

xlabel('time');
ylabel('neuron');
title('neural loading');


subplot(3, 3, [ 2 3])
x = ((1:size(HNew(1,:,tInd,factorInd),2)) - nLags )*binSize;
plot(x,squeeze(HNew(1,:,tInd,factorInd)));
chamod();

xlabel('time');
ylabel('loading');
%title('texture loading');
title( cdaData.fullRun.textures{useTList(tInd)});

xlim([min(x) max(x)])


subplot(3, 3, [ 5 6])
im = xMat(neuronOrder,:,tInd);
imagesc(x,1:141,im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');
chamod();


subplot(3, 3, [ 8 9])
im = reconXHat(WNew(:,1,1,factorInd,:), HNew(1,:,tInd,factorInd));
imagesc(x,1:141,im(neuronOrder,nLags+1:end-nLags) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');
chamod();

drawnow;
    
    
    
    
    
    
    
    if lastRun
        break
    end
end

clear X_shifted;
clear xHat_shifted;
clear XS;
clear dRdH;
clear dSmoothdH;
clear WTX;
clear WTXhat;
clear xHat2;

%xMat = xMat(:,nLags+1:end);
%xHat = xHat(:,nLags+1:end);
%H = H(:,nLags+1:end);

xMat = xMat(:,nLags+1:end-nLags,:);
xHat = xHat(:,nLags+1:end-nLags,:);
%H = H(:,nLags+1:end-nLags,:,:);

% Need to measure reconstruction error



%%



factorInd = 1;
tInd = 2;



[~,neuronOrder] = sort(nanmean(nanstd( xMat, [], 2),3));



clf;
subplot(6,3,1);
plot(iterCost);
axis([0 length(iterCost)+1 0 max(iterCost)*1.05]);


x = (1:size(WNew,5))*binSize;

subplot(6,3,[4 7]);
vals = squeeze(WNew(neuronOrder((end-3):end),1,1,factorInd,:))';
plot(x,vals);
axis([0 size(WNew,5)*binSize 0 max(vals(:)+eps)*1.05]);
chamod();

subplot(6,3,[10 13 16]);
y = 1:size(WNew,1);
im = squeeze(WNew(neuronOrder,1,1,factorInd,:));
imagesc([0 x],y,im);
%caxis([0 nanmean(im(:))*nanstd(im(:))*5]);
caxis([0 quantile(im(:), 0.9)]);


%caxis([-1 1]*nanstd(im(:))*1.2 + nanmean(im(:)));
chamod();

xlabel('time');
ylabel('neuron');
title('neural loading');


subplot(3, 3, [ 2 3])
x = ((1:size(HNew(1,:,tInd,factorInd),2)) - nLags )*binSize;
plot(x,squeeze(HNew(1,:,tInd,factorInd)));
chamod();

xlabel('time');
ylabel('loading');
%title('texture loading');
title( cdaData.fullRun.textures{useTList(tInd)});

xlim([min(x) max(x)])


subplot(3, 3, [ 5 6])
im = xMat(neuronOrder,:,tInd);
imagesc(x,1:141,im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');
chamod();


subplot(3, 3, [ 8 9])
im = reconXHat(WNew(:,1,1,factorInd,:), HNew(1,:,tInd,factorInd));
imagesc(x,1:141,im(neuronOrder,nLags+1:end-nLags) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');
chamod();








%% Take the existing Hs and Ws and fit Gaussians to them.

lambdaSigL1     = 10^1;
lambdaAL1       = 1;

nIterations     = 51;

nFindFactors    = 1;
nFactors        = nFindFactors;


aSmall      = 10^-4;
sigSmall    = 10^-4;

p = 0.4;

rotTInd = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];

useTList = [4 5 7 41];
%useTList = 5;
useTList = 1:59;

xMat        = permute( nanmean(fullPsth(:,:,useTList,:),4), [2 1 3] );

nNeurons    = size(xMat,1);
trueNT      = size(xMat,2);
nConditions = size(xMat,3);

nLags           = 249;

xMean = ones(nNeurons,nLags,nConditions).*nanmean(xMat,2);
xMat = cat(2,xMean,xMat,xMean);

useH = H(1,2:end-1,useTList,2);

[~, nT,~] = size(xMat);


smallNum    = max(xMat(:))*1e-6;




lagTimeVals = shiftdim(-floor(nLags/2):1:floor(nLags/2),-3).*binSize;
wMeanInit   = 0;
wSigmaInit  = 0.03;

uStep       = (wSigmaInit/3).^2;

%wMean       = ones(141,1).*wMeanInit;
%wSigma      = ones(141,1).*wSigmaInit;
%wA          = ones(141,1).*max(xMat(:));


wMean       = saveGaussParam(1,:)';
wSigma      = saveGaussParam(2,:)';
wA          = saveGaussParam(3,:)';


gaussFunc = @(t,mu,sigma)(exp( -(1/2).*((t-mu)./sigma).^2));


WGauss = wA.*repmat(gaussFunc(lagTimeVals, wMean,wSigma), [1 1,1,nFactors,1]);

tic
xHat = reconXHat(WGauss, useH); 
t = toc;
[0 t]


W = W.*median(median(xMat,2),3)./median(median(xHat,2),3);

tic
xHat = reconXHat(WGauss, useH); 
t = toc;
[0 t]

% Nice idea for cross-validation: 
% X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit


smoothKernel = ones(1,(2*nLags)-1);  % for factor competition
smallNum = max(xMat(:))*1e-6; 
lastTime = 0;

% Calculate initial cost
iterCost = zeros(nIterations+1, 1);
iterCost(1) = sqrt(mean((xMat(:)-xHat(:)).^2));

lastRun = false;
for iterInd = 1:nIterations
    iterInd
    
    tic
  
    WdiffMean   = (1./wSigma).*((lagTimeVals-wMean)./wSigma).*WGauss;
    xMeanHat    = reconXHat(WdiffMean, useH);
    uDelta      = sum(sum( (xMat-xHat).*xMeanHat,2),3).*uStep;
    
    if(any(isnan(uDelta(:))))
        1
        error()
    end
    
    WdiffSigma  = (1./wSigma).*(((lagTimeVals-wMean)./wSigma).^2).*WGauss;
    xSigmaHat   = reconXHat(WdiffSigma, useH); 
    sigDelta    = (sum(sum(xMat.*xSigmaHat,2),3)./(sum(sum(xHat.*xSigmaHat,2),3) + lambdaSigL1) + eps);

    if(any(isnan(sigDelta(:))))
        2
        error()
    end
    
    ADelta      = sum(sum(xMat.*xHat,2),3)./(sum(sum(xHat.*xHat,2),3) + lambdaAL1);
    
    if(any(isnan(ADelta(:))))
        3
        error()
    end
    
    
    if iterInd > 20
        'mean'
        wMean   = wMean + uDelta*p;
    end
    wSigma  = max(10^-3, wSigma.*(sigDelta*p + (1-p)));
    wA      = wA.*(ADelta*p+(1-p)) + aSmall;
    
    
    WGauss       = wA.*repmat(gaussFunc(lagTimeVals, wMean,wSigma), [1 1,1,nFactors,1]);

    if(any(isnan(WGauss(:))))
        4
        error()
    end
    

    clf;
    subplot(3,1,1);
    x = (1:size(useH(1,nLags+1:end-nLags,tInd,1),2))*binSize;
    plot(x,squeeze(useH(1,nLags+1:end-nLags,tInd,1)));
    chamod();
    title(iterInd);

    subplot(3,1,[2 3]);
    y = 1:size(WGauss,1);
    im = squeeze(WGauss(neuronOrder,1,1,1,:));
    %im = im./sum(im+eps,2);
    imagesc([0 x],y,im);
    caxis([0 quantile(im(:), 0.9)]);

    chamod();

    xlabel('time');
    ylabel('neuron');
    title('neural loading');
    drawnow;

    t = toc;
    [iterInd t]

    % Compute cost
    xHat = reconXHat(WGauss, useH); 
    
    t = toc;
    [iterInd t]

    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    if iterCost(iterInd+1) > iterCost(iterInd)
        Wgauss2 = WGauss.*median(median(xMat,2),3)./median(median(xHat,2),3);
        xHat2 = reconXHat(Wgauss2, useH); 
        iterCost2 = sqrt(mean((xMat(:)-xHat2(:)).^2));
        
        if iterCost2 < iterCost(iterInd+1)
            %W = W2;
            wA = wA.*median(median(xMat,2),3)./median(median(xHat,2),3);
            WGauss  = wA.*repmat(gaussFunc(lagTimeVals, wMean,wSigma), [1 1,1,nFactors,1]);

            xHat = xHat2;
            iterCost(iterInd+1) = iterCost2;
            
            clear xHat2;
            clear W2;
        end
        
        
    end
    t = toc;
    [iterInd t]
%     
%     if lastRun
%         break
%     end
end


clear X_shifted;
clear xHat_shifted;
clear XS;
clear dRdH;
clear dSmoothdH;
clear WTX;
clear WTXhat;


xMat = xMat(:,nLags+1:end-nLags,:);
xHat = xHat(:,nLags+1:end-nLags,:);
















%%


%% Next step analysis ideas

% Confine the Ws to a single peak - can we find a good shape for this?
% Smooth our parameters and try refitting the data
% 

%%

t= -1.5:0.001:1.5;

mu      = 0;
sigma   = 0.04;

f1 = exp( -(1/2).*((t-mu)./sigma).^2).*20;
f2 = ((t-mu)/sigma.^2).*exp( -(1/2).*((t-mu)./sigma).^2);
f3 = ((t-mu).^2/sigma.^3).*exp( -(1/2).*((t-mu)./sigma).^2);


clf;
hold all;
plot(t,f1);
plot(t,f2);
plot(t,f3);

xlim([-0.15 0.15]);

%%  New idea: confine W to be Gaussian



tolerance   = 10^-5;
lambda      = 10^-4.5;
lambda      = 0;
lambdaL1W   = 0; %10^-1;
lambdaHSmooth = 10.^1.6;
lambdaWSmooth = 10.^1.6;
lambdaSigL1     = 10^1;

nIterations     = 51;

nFindFactors    = 1;
nFactors        = nFindFactors+1;


aSmall = 10^-4;
sigSmall = 10^-4;



rotTInd = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];
%useTList = rotTInd;
%useTList = 1:10;
%useTList = 1:59;
useTList = [4 5 7 41];
useTList = 5;


%useInd = [1 2 3];

%xMat        = cityLightsTraces';
%xMat        = permute( meanFullPsth(:,:,[5 7]), [2 1 3] );
xMat        = permute( nanmean(fullPsth(:,:,useTList,:),4), [2 1 3] );
%xMat        = permute( meanFullPsth(:,:,[48]), [2 1 3] );

nNeurons    = size(xMat,1);
trueNT      = size(xMat,2);
nConditions = size(xMat,3);

nLags           = 249;

xMean = ones(nNeurons,nLags,nConditions).*nanmean(xMat,2);
xMat = cat(2,xMean,xMat,xMean);
%xMat = [zeros(nNeurons,nLags),xMat];

[~, nT,~] = size(xMat);


smallNum    = max(xMat(:))*1e-6;




lagTimeVals = shiftdim(-floor(nLags/2):1:floor(nLags/2),-3).*binSize;
wMeanInit   = 0;
wSigmaInit  = 0.03;

uStep       = (wSigmaInit/3).^2;

wMean       = ones(141,1).*wMeanInit;
wSigma      = ones(141,1).*wSigmaInit;
wA          = ones(141,1).*max(xMat(:));

gaussFunc = @(t,mu,sigma)(exp( -(1/2).*((t-mu)./sigma).^2));


W = wA.*repmat(gaussFunc(lagTimeVals, wMean,wSigma), [1 1,1,nFactors,1]);


% initialize factors
% W = max(xMat(:))*rand(nNeurons,1,1,nFactors,nLags); % K factors x N neurons x L lags. 
% midLag = floor(size(W,5)/2);
% W(:,1,1,2:end, 1:midLag) = W(:,1,1,2:end, 1:midLag) .* shiftdim(linspace(0.5,1,midLag), -3);
% W(:,1,1,2:end, midLag+1:end) = W(:,1,1,2:end, midLag+1:end) .* shiftdim(linspace(1,0.5,midLag), -3);

W(:,1,1,1,:) = repmat(nanmean(W(:,1,1,1,:),5), [1 1 1 1 size(W,5)]);

% eventually: K factors, T time points x C conditions. No need to be sparse.
% Why do we care about the Frobeneus norm of each row?
H = rand(1,nT,nConditions,nFactors); % normalize so frobenius norm of each row ~ 1
H(1,:,:,2:end) = H(1,:,:,2:end).*nanmean(xMat,1);
H(1,:,:,1) = repmat(nanmean(nanmean(H(1,:,:,1),2),3), [1 size(H,2) size(H,3) 1]);


norms = sqrt(sum(H.^2, 2));
H = H./norms;

tic
xHat = reconXHat(W, H); 
t = toc;
[0 t]


W = W.*median(median(xMat,2),3)./median(median(xHat,2),3);

tic
xHat = reconXHat(W, H); 
t = toc;
[0 t]

% Nice idea for cross-validation: 
% X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit


smoothKernel = ones(1,(2*nLags)-1);  % for factor competition
smallNum = max(xMat(:))*1e-6; 
lastTime = 0;

% Calculate initial cost
iterCost = zeros(nIterations+1, 1);
iterCost(1) = sqrt(mean((xMat(:)-xHat(:)).^2));

lastRun = false;
for iterInd = 1:nIterations
    iterInd
    % Are our updates still changing things? Or have we reached a local
    % minima?
    
%     if false && iterInd>5 && (iterCost(iterInd)+tolerance)>nanmean(iterCost((iterInd-5):iterInd-1))
%         [iterInd iterCost(iterInd) nanmean(iterCost((iterInd-5):iterInd-1))]
%         %cost = cost(1 : iter+1);  % trim vector
%         lastRun = true;
%         if iterInd>1
%             lambda = 0; % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
%         end
%     end
    tic
    
    % Normal update terms for convolutional non-negative matrix
    % factorization
    WTX = 0; %zeros(nFactors, nT);
    WTXhat = 0; %zeros(nFactors, nT);
    
    %xMatSum = sum(xMat,3);
    %xHatSum = sum(xHat,3);
    
    
    parfor lInd = 1 : nLags
        X_shifted = circshift(xMat,[0,-lInd+1,0]); 
        xHat_shifted = circshift(xHat,[0,-lInd+1,0]); 
        WTX = WTX + sum(W(:,1,1,:,lInd) .* X_shifted,1);
        WTXhat = WTXhat + sum(W(:,1,1,:,lInd) .* xHat_shifted,1);
    end
    t = toc;
    [iterInd t]
    
    % Compute regularization terms for H update
    if lambda>0
        dRdH = lambda.*permute(sum((shiftdim(~eye(nFactors),-3)).*convn(WTX, smoothKernel, 'same'),4),[1 2 3 5 4]);
    else 
        dRdH = zeros(1,1,1,size(H,4)); 
    end
    if false && lambdaL1H > 0
        
    else
        lambdaL1H = 0;
    end
    if false && params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH*(~eye(K))*conv2(H, smoothKernel, 'same');
    else
        dHHdH = 0;
    end
    
    if true
        %dSmoothdH = exp(lambdaHSmooth*[H(:,1)-H(:,2) (H(:,2:end-1)-(H(:,1:end-2)+H(:,3:end))/2) H(:,end)-H(:,end-1)]);
        %dSmoothdH = 1;
                
        dSmoothdH = lambdaHSmooth*cat(2, 2*H(:,1,:,:) - 3*H(:,1,:,:) + H(:,3,:,:), ...
            -3*H(:,1,:,:)+6*H(:,2,:,:)-4*H(:,3,:,:)+H(:,4,:,:), ...
            H(:,1:end-4,:,:) - 4*H(:,2:end-3,:,:) + 6*H(:,3:end-2,:,:) - 4*H(:,4:end-1,:,:) + H(:,5:end,:,:), ...
            H(:,1,:,:)-4*H(:,2,:,:)+6*H(:,end-1,:,:)-3*H(:,end,:,:), ...
            H(:,end-2,:,:) - 3*H(:,end-1,:,:) + 2*H(:,end,:,:));
        
        dSmoothdH(dSmoothdH < 0) = 0;
            
    else
        dSmoothdH = 0;
    end
    %dRdH = dRdH + lambdaL1H + dHHdH; % include L1 sparsity, if specified
    dRdH = dRdH + dSmoothdH;
   
    
                
    if(any(dSmoothdH(:) < 0))
        error()
    end

    
    % Update H
    denom = (WTXhat + dRdH +eps);
    if(any(denom(:) < 0))
        error()
    end    
    
    H(1,:,:,1) = H(1,:,:,1) .* nanmean(WTX(1,:,:,1) ./ denom(1,:,:,1),2); % How do we calculate all of these?
    H(1,:,:,2:end) = H(1,:,:,2:end) .* WTX(1,:,:,2:end) ./ denom(1,:,:,2:end); % How do we calculate all of these?
        
%     norms = sqrt(sum(H.^2, 2))';
%     H = H./norms';
%     W = W.*norms;
    
    if false % parameter shifting 
        [W, H] = shiftMyFactors(W, H);  
        W = W+smallNum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    


    
    
    %H = diag(1 ./ (norms+eps)) * H;
    %for lInd = 1 : nLags
    %    W(:, :, lInd) = W(:, :, lInd) * diag(norms);
    %end 
    
    
    
    if lambda>0  %  && params.useWupdate
        XS = convn(xMat, smoothKernel, 'same'); 
    end
    
    if false
        %dSmoothdW = exp(lambdaWSmooth.*cat(3,W(:,:,1)-W(:,:,2),(W(:,:,2:end-1)-(W(:,:,1:end-2)+W(:,:,3:end))/2),W(:,:,end)-W(:,:,end-1)));
        
        dSmoothdW = lambdaWSmooth*cat(5, 2*W(:,:,:,:,1) - 3*W(:,:,:,:,1) + W(:,:,:,:,3), ...
            -3*W(:,:,:,:,1)+6*W(:,:,:,:,2)-4*W(:,:,:,:,3)+W(:,:,:,:,4), ...
            W(:,:,:,:,1:end-4) - 4*W(:,:,:,:,2:end-3) + 6*W(:,:,:,:,3:end-2) - 4*W(:,:,:,:,4:end-1) + W(:,:,:,:,5:end), ...
            W(:,:,:,:,1)-4*W(:,:,:,:,2)+6*W(:,:,:,:,end-1)-3*W(:,:,:,:,end), ...
            W(:,:,:,:,end-2) - 3*W(:,:,:,:,end-1) + 2*W(:,:,:,:,end));
            
                       
        dSmoothdW(dSmoothdW < 0) = 0;
        
    else
        dSmoothdW = 0;
    end
    
%     wMean       = wMeanInit;
%     wSigma      = wSigmaInit;
%     wA          = ones(141,1).*max(xMat(:));
%     
%     lagTimeVals
%     
%     f1 = exp( -(1/2).*((t-mu)./sigma).^2);
%     f2 = ((t-mu)/sigma.^2).*exp( -(1/2).*((t-mu)./sigma).^2);
%     f3 = ((t-mu).^2/sigma.^3).*exp( -(1/2).*((t-mu)./sigma).^2);

    
    WdiffMean   = (1./wSigma).*((lagTimeVals-wMean)./wSigma).*W;
    xMeanHat    = reconXHat(WdiffMean, H);
    uDelta      = sum(sum( (xMat-xHat).*xMeanHat,2),3).*uStep;
    
    if(any(isnan(uDelta(:))))
        1
        error()
    end
    
    WdiffSigma  = (1./wSigma).*(((lagTimeVals-wMean)./wSigma).^2).*W;
    xSigmaHat   = reconXHat(WdiffSigma, H); 
    sigDelta    = (sum(sum(xMat.*xSigmaHat,2),3)./(sum(sum(xHat.*xSigmaHat,2),3) + lambdaSigL1) + eps);

    if(any(isnan(sigDelta(:))))
        2
        error()
    end
    
    ADelta      = sum(sum(xMat.*xHat,2),3)./(sum(sum(xHat.*xHat,2),3) + lambdaAL1);
    
    if(any(isnan(ADelta(:))))
        3
        error()
    end
    
    
    wMean   = wMean + uDelta;
    wSigma  = wSigma.*sigDelta + sigSmall;
    wA      = wA.*ADelta + aSmall;
    
    W       = wA.*repmat(gaussFunc(lagTimeVals, wMean,wSigma), [1 1,1,nFactors,1]);

    if(any(isnan(W(:))))
        4
        error()
    end
    
% 
% clf;
% subplot(3,1,1);
% x = (1:size(H(1,nLags+1:end-nLags,tInd,factorInd),2))*binSize;
% plot(x,squeeze(H(1,nLags+1:end-nLags,tInd,factorInd)));
% chamod();
% 
% subplot(3,1,[2 3]);
% y = 1:size(W,1);
% im = squeeze(W(neuronOrder,1,1,factorInd,:));
% imagesc([0 x],y,im);
% caxis([0 quantile(im(:), 0.9)]);
% 
% chamod();
% 
% xlabel('time');
% ylabel('neuron');
% title('neural loading');
% drawnow;

    
%     % Update each W at each lag separately
%     parfor lInd = 1:nLags
%         H_shifted = circshift(H,[0,lInd-1,0,0]);
%         XHT = sum(sum(xMat .* H_shifted,2),3);
%         XhatHT = sum(sum(xHat .* H_shifted,2),3);
%         
%         
%         if lambda>0 % && params.useWupdate    % Often get similar results with just H update, so option to skip W update
%             dRdW = lambda.*permute(sum(sum(sum(XS.*H_shifted,2),3).*shiftdim(~eye(nFactors),-3),4), [1 2 3 5 4]); 
%         else
%             dRdW = 0;
%         end
% %         if ~isempty(lambdaL1W)
% %             
% %         else
% %             lambdaL1W = 0;
% %         end
% %         if false %&& params.lambdaOrthoW>0
% %             %dWWdW = params.lambdaOrthoW*Wflat*(~eye(K));
% %         else
% %             dWWdW = 0;
% %         end
% 
%         %dRdW = dRdW + lambdaL1W + dWWdW; % include L1 and Worthogonality sparsity, if specified
%         %dRdW = dRdW.*dSmoothdW(:,:,lInd);
% 
%         
%         if lInd == 1 
%             if iterInd == 5
%                 %error()
%             end
%         end
%         
%         % Update W
%         %W(:,1,1,:,lInd) = W(:,1,1,:,lInd) .* XHT ./ (XhatHT + dRdW + eps); % How do we calculate all of these?
%         denom = (XhatHT + dRdW + dSmoothdW(:,:,:,:,lInd) + eps);
%         if(any(denom(:) < 0))
%             error()
%         end   
%         W(:,1,1,:,lInd) = W(:,1,1,:,lInd) .* XHT ./ denom; 
%         
%     end
    t = toc;
    [iterInd t]
    W(:,1,1,1,:) = repmat(nanmean(W(:,1,1,1,:),5), [1 1 1 1 size(W,5)]);
    
    
    % Compute cost
    xHat = reconXHat(W, H); 
    
    t = toc;
    [iterInd t]
    
    % mask = find(params.M == 0); % find masked (held-out) indices 
    % X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    
    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    if iterCost(iterInd+1) > iterCost(iterInd)
        W2 = W.*median(median(xMat,2),3)./median(median(xHat,2),3);
        xHat2 = reconXHat(W2, H); 
        iterCost2 = sqrt(mean((xMat(:)-xHat2(:)).^2));
        
        if iterCost2 < iterCost(iterInd+1)
            %W = W2;
            wA = wA.*median(median(xMat,2),3)./median(median(xHat,2),3);
            W       = wA.*repmat(gaussFunc(lagTimeVals, wMean,wSigma), [1 1,1,nFactors,1]);

            xHat = xHat2;
            iterCost(iterInd+1) = iterCost2;
            
            clear xHat2;
            clear W2;
        end
        
        
    end
    t = toc;
    [iterInd t]
    
    if lastRun
        break
    end
end


clear X_shifted;
clear xHat_shifted;
clear XS;
clear dRdH;
clear dSmoothdH;
clear WTX;
clear WTXhat;

%xMat = xMat(:,nLags+1:end);
%xHat = xHat(:,nLags+1:end);
%H = H(:,nLags+1:end);

xMat = xMat(:,nLags+1:end-nLags,:);
xHat = xHat(:,nLags+1:end-nLags,:);
%H = H(:,nLags+1:end-nLags,:,:);

% Need to measure reconstruction error

%


factorInd = 2;
tInd = 1;



[~,neuronOrder] = sort(nanmean(nanstd( xMat, [], 2),3));



clf;
subplot(6,3,1);
plot(iterCost);
axis([0 length(iterCost)+1 0 max(iterCost)*1.05]);


x = (1:size(W,5))*binSize;

subplot(6,3,[4 7]);
vals = squeeze(W(neuronOrder((end-3):end),1,1,factorInd,:))';
plot(x,vals);
axis([0 size(W,5)*binSize 0 max(vals(:)+eps)*1.05]);
chamod();

subplot(6,3,[10 13 16]);
y = 1:size(W,1);
im = squeeze(W(neuronOrder,1,1,factorInd,:));
imagesc([0 x],y,im);
%caxis([0 nanmean(im(:))*nanstd(im(:))*5]);
caxis([0 quantile(im(:), 0.9)]);


%caxis([-1 1]*nanstd(im(:))*1.2 + nanmean(im(:)));
chamod();

xlabel('time');
ylabel('neuron');
title('neural loading');


subplot(3, 3, [ 2 3])
x = (1:size(H(1,nLags+1:end-nLags,tInd,factorInd),2))*binSize;
plot(x,squeeze(H(1,nLags+1:end-nLags,tInd,factorInd)));
chamod();

xlabel('time');
ylabel('loading');
%title('texture loading');
title( cdaData.fullRun.textures{useTList(tInd)});



subplot(3, 3, [ 5 6])
im = xMat(neuronOrder,:,tInd);
imagesc(x,1:141,im );
%caxis([0 nanmean(im(:))*nanstd(im(:))*50]);
caxis([0 quantile(im(:), 0.9)]);

title('actual data');
chamod();


subplot(3, 3, [ 8 9])
im = reconXHat(W(:,1,1,factorInd,:), H(1,:,tInd,factorInd));
imagesc(x,1:141,im(neuronOrder,nLags+1:end-nLags) );
%caxis([0 nanmean(im(:))*nanstd(im(:))*500]);
caxis([0 quantile(im(:), 0.9)]);

title('predicted data');
chamod();