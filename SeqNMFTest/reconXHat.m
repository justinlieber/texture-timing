function X_hat = reconXHat(W,H)
% ------------------------------------------------------------------------
% USAGE: X_hat = helper.reconstruct(W,H)
% ------------------------------------------------------------------------
% INPUTS
% W:      W is a NxKxL tensor which gives the neuron basis
%         functions which are used for the reconstructions. The L'th NxK slice
%         of W is the neural basis set for a lag of L.
%
% H:      H is a KxT matrix which gives timecourses for each factor 
% ------------------------------------------------------------------------
% OUTPUTS
% X_hat:  The reconstruction X_hat = W (*) H; 
% ------------------------------------------------------------------------
% originally: Emily Mackevicius and Andrew Bahle
% edited by: JDL

[N,~,~,K,L] = size(W);
[~,T,C,K2] = size(H);

% zeropad by L
hMean = ones(1,L,C,K).*nanmean(H,2);
H = cat(2,hMean,H,hMean);
T = T+2*L;
X_hat = zeros(N,T,C);

parfor tau = 1:L % go through every offset from 1:tau
    X_hat = X_hat + sum(W(:,1,1,:,tau) .* circshift(H(1,:,:,:),tau-1,2),4);
    %X_hat = X_hat + W(:, :, tau) * circshift(H,[0,tau-1]);
end

% undo zer0padding
X_hat = X_hat(:,(L+1):(end-L),:);

end

% %%
% tau = 150;
% x = sum(W(:,1,1,:,tau) .* circshift(H(1,:,:,:),tau-1,2),4);
% 
% clf;
% hold all;
% plot(H(1,(L+1):(end-L),1,2)*0.2)
% plot(x(20,(L+1):(end-L),1)*30)
% plot(X_hat(20,(L+1):(end-L),1))
% %%