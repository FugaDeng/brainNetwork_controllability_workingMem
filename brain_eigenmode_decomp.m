% Beynel, Deng, Davis, 7/2020
% eigenmode analysis in Figure 3, Beynel et al. (2020) JNeuro
%% data definition

subj=readtable('AllSubjData.csv');
gSC=zeros(471,471,size(subj,1));% Structural network of all subjects
gUVA=zeros(471,4,size(subj,1)); % Univariate activation vectors (brain states), region-by-condition-by-subject


dpath='Data/';
for i=1:size(subj,1)
    % load structural network for each participant.
    % each structural network is a 471-by-471 matrix
    gSC(:,:,i)=csvread([dpath 'SC_ALL/' num2str(subj.ID(i)) 'streamline.csv']).*(eye(471)==0);
    
    
    % import condition-specific brain states.
    % the brain state under each condition is a 471-by-1 vector of brain
    % activation
    gUVA(:,1,i)=dlmread([dpath 'UVAR_ALL/4diffHOA471Beta/'...
        num2str(subj.ID(i)) '/COPEdiff1.txt']);% easiest condition
    gUVA(:,2,i)=dlmread([dpath 'UVAR_ALL/4diffHOA471Beta/'...
        num2str(subj.ID(i)) '/COPEdiff2.txt']);
    gUVA(:,3,i)=dlmread([dpath 'UVAR_ALL/4diffHOA471Beta/'...
        num2str(subj.ID(i)) '/COPEdiff3.txt']);
    gUVA(:,4,i)=dlmread([dpath 'UVAR_ALL/4diffHOA471Beta/'...
        num2str(subj.ID(i)) '/COPEdiff4.txt']);% hardest condition
end

%% decomposing brain states into eigenmodes

Loading=zeros(4,471,size(subj,1)); % data for Figure 3A
% the first dimension: task condition (difficulty level)
% the second dimension: loading onto the eigenmodes
% the third dimension: number of participants

allSubjModes=zeros(471,471,23);% data for Figure 3B

for i=1:size(subj,1)
    % tmpvec: eigen vectors, i.e. eigenmodes of activation
    [tmpvec,tmplambda]=eig(gSC(:,:,i)); 
    
    % The following three lines sort the eigenmodes according to the stability of the modes
    % stability is reflected by the eigenmode's associated eigenvalue
    % small eigen value: less stable, hard to reach
    % large eigen value: more stable, easy to reach
    tmplambda=diag(tmplambda); 
    [tmplambda,tmpind]=sort( abs(tmplambda),'ascend' ); 
    tmpvec=tmpvec(:,tmpind);
    
    
    allSubjModes(:,:,i)=tmpvec; 
    
    % computing the loading of actual activation onto the eigenmodes
    % by computing the inner product between the two set of vectors
    Loading(:,:,i)=(gUVA(:,:,i)'*tmpvec).^2;
    
    % normalization
    Loading(1,:,i)=Loading(1,:,i)/sum(gUVA(:,1,i).^2);
    Loading(2,:,i)=Loading(2,:,i)/sum(gUVA(:,2,i).^2);
    Loading(3,:,i)=Loading(3,:,i)/sum(gUVA(:,3,i).^2);
    Loading(4,:,i)=Loading(4,:,i)/sum(gUVA(:,4,i).^2);
end

