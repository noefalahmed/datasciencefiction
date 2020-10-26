
%   PCA applied to alphabets (images).

%   Goal: Reducing dimensionalities of images by finding important information and discarding the rest.











datadir='.';    % directory where the data files reside
dataset={'arial','bookman_old_style','century','comic_sans_ms','courier_new',...
  'fixed_sys','georgia','microsoft_sans_serif','palatino_linotype',...
  'shruti','tahoma','times_new_roman'};
datachar='abcdefghijklmnopqrstuvwxyz';

Rows=64;    % all images are 64x64
Cols=64;
n=length(dataset)*length(datachar);  % total number of images
p=Rows*Cols;   % number of pixels

X=zeros(p,n);  % images arranged in columns of X
k=1;
for dset=dataset
for ch=datachar
  fname=sprintf('%s/%s/%s.tif',datadir,char(dset),ch);
  img=imread(fname);
  X(:,k)=reshape(img,1,Rows*Cols);
  k=k+1;
end
end

%return

%display ('samples of the training data');
for k=1:length(dataset)
  img=reshape(X(:,26*(k-1)+1),64,64);
  figure(1); subplot(3,4,k); image(img); 
  axis('image'); colormap(gray(256)); 
  title(dataset{k},'Interpreter','none');
end
m=mean(X);
R=double((1)./(n-1))*((X-m)*(X-m)');
[U,S,V]=svd(R,0);
ems=U(1:4096,1:12);
e=10;
selecedEV=U(1:4096,1:e);
encod=(selecedEV')*((X-m));
for i=1:length(dataset)
  img=reshape(ems(:,i),64,64);
  figure(2); subplot(3,4,i); imagesc(img); 
  axis('image'); colormap(gray(256)); 
  title(dataset{i},'Interpreter','none');
end
figure(3);plot(encod(:,1:e));
decode=selecedEV*encod;

for i=1:length(dataset)
  img=reshape(decode(:,26*(i-1)+1),64,64);
  figure(4); subplot(3,4,i); image(img); 
  axis('image'); colormap(gray(256)); 
  title(dataset{i},'Interpreter','none');
end
