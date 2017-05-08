function data_sn=speaker_norm(data,speakerIDs)
% real speaker IDs for normalization
speaker_list=unique(speakerIDs');

cnt=0;
data_sn=zeros(size(data));
stats_clust=cell(numel(speaker_list),2);
filter_feats=std(data)==0;
for cnt=1:numel(speaker_list)
    spkrid=speaker_list(cnt);
    spkr_filter=find(speakerIDs==spkrid);
    stats_clust{cnt}.mean=mean(data(spkr_filter,:));
    stats_clust{cnt}.std=std(data(spkr_filter,:));
    filter_feats=filter_feats|stats_clust{cnt}.std==0;
    %all_data_fisher_sn(spkr_filter,:)=scal(all_data_fisher(spkr_filter,:),stats_clust{cnt}.mean,stats_clust{cnt}.std);
end
filter_feats=~filter_feats;

for cnt=1:numel(speaker_list)
    spkrid=speaker_list(cnt);
    spkr_filter=find(speakerIDs==spkrid);

    data_sn(spkr_filter,filter_feats)=scal(data(spkr_filter,filter_feats),stats_clust{cnt}.mean(filter_feats),stats_clust{cnt}.std(filter_feats));
    %gaussianize(all_data_fisher(spkr_filter,filter_feats));
end
disp('Data Speaker Normalized')