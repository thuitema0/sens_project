%% x_full_simulation.m
% Full pipeline: load DICOM, extract tumour mask, visualize, simulate COM displacement

%% 0. User parameters (edit as needed)
patientID          = 'sens_001';    % e.g. 'sens_001'
baseDir            = fullfile(patientID, '7368');
fixed_vox          = [ ...            % fixed fiducials in voxel space (Nx3)
    206,198,264;
    226,194,266;
    252,186,278;
    272,194,264;
    296,194,262;
    248,154,242;
    226,186,234;
    274,186,232;
    244,170,228;
    212,184,212;
    286,182,210;
    248,162,222;
    248,172,208;
    248,162,188 ];
meanNoise          = 0;              % mean of Gaussian noise in mm
sigmaNoise         = 1;              % sigma of Gaussian noise in mm
numIter            = 100;            % number of Monte Carlo iterations
view_scan          = false;           % plot CT + tumour before simulation
check_registration = true;           % plot example registration
scale              = 0.5;            % downsampling factor for visualization

%% 1. Locate DICOM folders
% Images folder
d = dir(fullfile(baseDir,'Images','*'));
isSubdir = [d.isdir] & ~ismember({d.name},{'.','..'});
if ~any(isSubdir)
    error('No subfolder under Images');
end
ctDir = fullfile(baseDir,'Images', d(find(isSubdir,1)).name);

% StructureSets folder
d2 = dir(fullfile(baseDir,'StructureSets','*'));
isSubdir2 = [d2.isdir] & ~ismember({d2.name},{'.','..'});
if ~any(isSubdir2)
    error('No subfolder under StructureSets');
end
ssDir = fullfile(baseDir,'StructureSets', d2(find(isSubdir2,1)).name);

%% 2. Read CT volume + spacing
ctFiles = dir(fullfile(ctDir,'*.dcm'));
[ctVol, ~] = dicomreadVolume(ctDir);
ctVol = squeeze(ctVol);
info0 = dicominfo(fullfile(ctDir,ctFiles(1).name));
dx = info0.PixelSpacing(1);
dy = info0.PixelSpacing(2);
if isfield(info0,'SpacingBetweenSlices'), dz = info0.SpacingBetweenSlices; else dz = info0.SliceThickness; end
spacing = [dx dy dz];

%% 3. Read & sort slice positions
nSlices = numel(ctFiles);
infoSlices = cell(nSlices,1);
zPos = zeros(nSlices,1);
for i = 1:nSlices
    infoSlices{i} = dicominfo(fullfile(ctDir,ctFiles(i).name));
    zPos(i) = infoSlices{i}.ImagePositionPatient(3);
end
[sortedZ, sortIdx] = sort(zPos);
origX0 = infoSlices{sortIdx(1)}.ImagePositionPatient(1);
origY0 = infoSlices{sortIdx(1)}.ImagePositionPatient(2);

%% 4. Extract ROI names & tumour mask
ssFiles = dir(fullfile(ssDir,'*.dcm'));
rtstruct = dicominfo(fullfile(ssDir,ssFiles(1).name));
rois   = rtstruct.StructureSetROISequence;
conts  = rtstruct.ROIContourSequence;
namesROI = fieldnames(rois);
roiNames = cellfun(@(f) rois.(f).ROIName, namesROI, 'UniformOutput',false);
matchIdx = find(contains(lower(roiNames),'htv') | contains(lower(roiNames),'tumour'));
if isempty(matchIdx)
    fprintf('Available ROIs:'); fprintf('  %s', roiNames{:});
    matchIdx = input('Select ROI index: ');
end
names    = fieldnames(conts);              % get a cell array of all contour fields
itemName = names{matchIdx};                % pick the one you want
contSeq = conts.(itemName).ContourSequence;
nContours = numel(fieldnames(contSeq));
[nX,nY,nZ] = size(ctVol);
maskTum = false(nY,nX,nZ);
for c = 1:nContours
    pts = reshape(contSeq.(sprintf('Item_%d',c)).ContourData,3,[])';
    [~,sliceZ] = min(abs(sortedZ - pts(1,3)));
    xPix = round((pts(:,1)-origX0)/dx)+1;
    yPix = round((pts(:,2)-origY0)/dy)+1;
    maskTum(:,:,sliceZ) = maskTum(:,:,sliceZ) | poly2mask(xPix,yPix,nY,nX);
end
maskTum = smooth3(maskTum,'box',5);

%% Prepare downsampled volumes for visualization
ctVolSmall   = imresize3(ctVol,scale,'linear');
maskTumSmall = imresize3(maskTum,scale,'nearest');
spacingSmall = spacing ./ scale;

%% Optional 3D visualization
if view_scan
    fvCT  = isosurface(ctVolSmall,300);
    fvTum = isosurface(maskTumSmall,0.5);
    figure('Name','Initial CT + Tumour'); hold on;
    p1 = patch(fvCT); isonormals(ctVolSmall,p1);
    p1.FaceColor=[0.8 0.8 0.8]; p1.FaceAlpha=0.2; p1.EdgeColor='none';
    p2 = patch(fvTum); isonormals(maskTumSmall,p2);
    p2.FaceColor='red'; p2.FaceAlpha=0.6; p2.EdgeColor='none';
    daspect(spacingSmall); view(3); camlight; lighting gouraud;
    title('CT + Tumour Segmentation');
end

%% Example registration plot
if check_registration
    fixed_real = fixed_vox .* spacing;
    noisy_real = fixed_real + meanNoise + sigmaNoise*randn(size(fixed_vox));
    [~,~,tr] = procrustes(fixed_real,noisy_real,'Scaling',false);
    R = tr.T; t = tr.c(1,:);
    Tvox2real = diag([spacing 1]); Treal2vox = diag([1./spacing 1]);
    Treg      = eye(4); Treg(1:3,1:3)=R; Treg(4,1:3)=t;
    Tcomp     = Treal2vox * Treg * Tvox2real;
    tform     = affine3d(Tcomp);
    ref3      = imref3d(size(maskTum),dx,dy,dz);
    warped    = imwarp(maskTum,tform,'OutputView',ref3,'Interp','nearest');
    % Downsample for consistency
    warpedSmall = imresize3(warped,scale,'nearest');
    % plot
    figure('Name','Example Registration'); hold on;
    pC = patch(isosurface(ctVolSmall,300)); isonormals(ctVolSmall,pC);
    pC.FaceColor=[0.8 0.8 0.8]; pC.FaceAlpha=0.1; pC.EdgeColor='none';
    pO = patch(isosurface(maskTumSmall,0.5)); isonormals(maskTumSmall,pO);
    pO.FaceColor='blue'; pO.FaceAlpha=0.4; pO.EdgeColor='none';
    pW = patch(isosurface(warpedSmall,0.5)); isonormals(warpedSmall,pW);
    pW.FaceColor='red'; pW.FaceAlpha=0.4; pW.EdgeColor='none';
    plot3(fixed_real(:,1),fixed_real(:,2),fixed_real(:,3),'g*','MarkerSize',8);
    plot3(noisy_real(:,1),noisy_real(:,2),noisy_real(:,3),'ro','MarkerSize',6);
    daspect(spacingSmall); view(3); camlight; lighting gouraud;
    legend('CT','Orig Tumour','Warped Tumour','Fixed pts','Noisy pts');
    title('Debug: One Registration Example');
end

%% Monte Carlo COM displacement & CSV
fixed_real = fixed_vox .* spacing;
outName    = sprintf('%s_noise%g_%g.csv',patientID,meanNoise,sigmaNoise);
fid        = fopen(outName,'w'); fprintf(fid,'Iteration,COM_disp_mm');
for i = 1:numIter
    noisy_real = fixed_real + meanNoise + sigmaNoise*randn(size(fixed_vox));
    [~,~,tr] = procrustes(fixed_real,noisy_real,'Scaling',false);
    R = tr.T; t = tr.c(1,:);
    Tvox2real = diag([spacing 1]); Treal2vox = diag([1./spacing 1]);
    Treg      = eye(4); Treg(1:3,1:3)=R; Treg(4,1:3)=t;
    Tcomp     = Treal2vox * Treg * Tvox2real;
    tform     = affine3d(Tcomp);
    ref3      = imref3d(size(maskTum),dx,dy,dz);
    warped    = imwarp(maskTum,tform,'OutputView',ref3,'Interp','nearest');
    C0 = regionprops3(maskTum,   'Centroid'); C0 = C0.Centroid(1,:);
    C1 = regionprops3(warped,    'Centroid'); C1 = C1.Centroid(1,:);
    dv = (C1 - C0) .* spacing; dispVal = norm(dv);
    fprintf(fid,'%d,%.4f',i,dispVal);
end
fclose(fid);
fprintf('Saved COM results to %s', outName);
