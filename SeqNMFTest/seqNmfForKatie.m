
tolerance       = 10^-5;
lambda          = 0;
lambdaL1W       = 10^-1;
lambdaHSmooth   = 10.^1.6;
lambdaWSmooth   = 10.^1.6;

nIterations     = 251;

nFindFactors    = 1;
nFactors        = nFindFactors;


periodicTList = [5 7 9 10 11 13 14 23 24 35 39 47 48 52 53 54];
%useTList = periodicTList;
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
W(:,1,1,:, 1:midLag) = W(:,1,1,2:end, 1:midLag) .* shiftdim(linspace(0.5,1,midLag), -3);
W(:,1,1,:, midLag+1:end) = W(:,1,1,2:end, midLag+1:end) .* shiftdim(linspace(1,0.5,midLag), -3);

%W(:,1,1,:,[1 end]) = 0;


% eventually: K factors, T time points x C conditions. No need to be sparse.
% Why do we care about the Frobeneus norm of each row?
H = rand(1,nT,nConditions,nFactors); % normalize so frobenius norm of each row ~ 1
H(1,:,:,:) = H(1,:,:,2:end).*nanmean(xMat,1);


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
    
    H(1,:,:,:) = H(1,:,:,2:end) .* WTX(1,:,:,2:end) ./ denom(1,:,:,2:end); % How do we calculate all of these?
        
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
    
    
    % Compute cost
    xHat = reconXHat(W, H); 
    
    t = toc;
    [iterInd t]
    
    % mask = find(params.M == 0); % find masked (held-out) indices 
    % X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    
    iterCost(iterInd+1) = sqrt(mean((xMat(:)-xHat(:)).^2));

    % Quick updating step to prevent outliers
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

%% Plot factors

factorInd = 1;
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