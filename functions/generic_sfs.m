function [z,N,XYZ,N_SH] = generic_sfs(data,params,options);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Preprocessings
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Subsampling
	if(options.ratio>1)
		data.I = data.I(1:options.ratio:end,1:options.ratio:end,:);
		data.mask = data.mask(1:options.ratio:end,1:options.ratio:end);
		data.mask_z0 = data.mask_z0(1:options.ratio:end,1:options.ratio:end);
		data.rho = data.rho(1:options.ratio:end,1:options.ratio:end,:);
		data.z0 = data.z0(1:options.ratio:end,1:options.ratio:end);
		data.z_init = data.z_init(1:options.ratio:end,1:options.ratio:end);
		data.K(1:2,:) = data.K(1:2,:)./options.ratio;
	end
	clear data.ratio;
	
	% Initialization of depth
	z = data.z_init;
	
	% Masked pixels
	imask = find(data.mask>0);
	imask_z0 = find(data.mask_z0>0);
	[foo,imask_z0] = ismember(imask_z0,imask);
	imask_z0 = imask_z0(find(foo>0));

	clear data.mask_z0 data.mask

	% Auxiliary variables
	npix = length(imask);
	[nrows,ncols,nchannels] = size(data.I);
	if(data.K(1,3)>0)
		[xx,yy] = meshgrid(1:ncols,1:nrows);
		xx = xx(imask);
		xx = xx-data.K(1,3);
		yy = yy(imask);
		yy = yy-data.K(2,3);
		z = log(z(imask));
		data.z0 = log(data.z0(imask));
	else
		xx = zeros(npix,1);
		yy = zeros(npix,1);
		data.K(1,1) = 1;
		data.K(2,2) = 1;
		data.K(1,3) = 0;
		z = z(imask);
		data.z0 = data.z0(imask);
	end
	G = make_gradient(data.mask); % Finite differences stencils
	Dx = G(1:2:end-1,:);
	Dy = G(2:2:end,:);
	clear G

	% Set BFGS options

	clear options.maxit_bfgs options.tolX_bfgs options.tolFun_bfgs options.check_grad options.use_jac

	% Vectorize data
	data.I = reshape(data.I,[nrows * ncols,nchannels]);
	data.I = data.I(imask,:);
	data.rho = reshape(data.rho,[nrows * ncols,nchannels]);
	data.rho = data.rho(imask,:);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Initialization
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% Initial gradient
	zx = Dx*z;
	zy = Dy*z;
	zx(isnan(zx)) = 0;
	zy(isnan(zy)) = 0;
	z(isnan(z)) = 1;

	% Initial augmented normals
	N = zeros(npix,9); 
	dz = max(eps,sqrt((data.K(1,1)*zx).^2+(data.K(2,2)*zy).^2+(-1-xx.*zx-yy.*zy).^2));
	N(:,1) = data.K(1,1)*zx./dz;
	N(:,2) = data.K(2,2)*zy./dz;
	N(:,3) = (-1-xx.*zx-yy.*zy)./dz;
	N(:,4) = 1;
	N(:,5) = N(:,1).*N(:,2);
	N(:,6) = N(:,1).*N(:,3);
	N(:,7) = N(:,2).*N(:,3);
	N(:,8) = N(:,1).^2-N(:,2).^2;
	N(:,9) = 3*N(:,3).^2-1;

	% Initial A and b fields
	A = zeros(npix,nchannels,2); 
	B = zeros(npix,nchannels,1);
	% data.rho shape [ncols*nrows,3]
	% data.K(1,1) [1 1]
	% data.s(ch,1) [1 1]
	% xx [nrows*ncols,1]
	% data.s(1,3) [1 1]
	% dz [nrows*ncols 1]
	%
	for ch = 1:nchannels
		A(:,ch,1) = data.rho(:,ch).*(data.K(1,1)*data.s(ch,1)-xx*data.s(ch,3))./dz;
		A(:,ch,2) = data.rho(:,ch).*(data.K(2,2)*data.s(ch,2)-yy*data.s(ch,3))./dz;

		B(:,ch) = data.I(:,ch)+data.rho(:,ch)*data.s(ch,3)./dz-data.rho(:,ch).*sum(bsxfun(@times,data.s(ch,4:9),N(:,4:9)),2);
	end
    
	
	% Initial energy: shading + prior + smoothness
	energy = 0;
	% Shading term
	for ch = 1:nchannels
		energy = energy + 0.5*params.lambda*sum((A(:,ch,1).*zx+A(:,ch,2).*zy-B(:,ch)).^2);
	end
	% Prior term
	energy = energy + 0.5*params.mu*sum((z(imask_z0)-data.z0(imask_z0)).^2);
	% Smoothness term
	energy = energy+params.nu*sum(dz);