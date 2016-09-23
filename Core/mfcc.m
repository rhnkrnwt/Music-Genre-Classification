function [ FINAL ] = mfcc(DIR, fname)

% get audio sample
cd(DIR);
[Y, Fs] = audioread(fname);
cd ..
% ------------------ %

% initialise variables
totalSeconds = 25; % consider first < ** > seconds of sample from the 30ish seconds to avoid errors with shorter samples
startTime = 8; % obtain middle 50% : 15 seconds
endTime = 23;
nCol = 750; % number of frames
nRow = 441; % number of samples in a frame
nhRow = 220; % half
plots = 20; % number of Mel bins
lT = 30; % lower threshold of human hearing
uT = 6000; % upper threshold
% ------------------ %

% MFCC : Mel Frequency Cepstral Coefficients Algorithms
Y = Y(1:Fs*totalSeconds,:);
X = Y(startTime*Fs:endTime*Fs,:); % taking middle 50% of the sample
n = Fs / 50; % number of samples in a frame of 20ms = 1000/20
S = zeros(n,nCol); % 750 is the number of frames
for k = 0:nCol-1
    S(:, k+1) = X(1+n*k:n*(k+1),:);
end 

H = repmat(hamming(nRow),[1, nCol]); % hamming window for signal smoothening
smooth_S = S .* H;

Si = zeros(size(smooth_S));
for k = 1:nCol
    Si(:,k) = fft(smooth_S(:,k)); % take dft
end

P = (abs(Si( 1:nhRow , :)).^2)./nRow; %periodogram based power spectrum

% Values from www.psbspeakers.com/Images/Audiotopics/fChart.gif
lower = mc(lT);
upper = mc(uT);
step = (upper - lower) / plots+1;
Mel = (linspace(0,plots+1,plots+2)')*step;
Mel = Mel + lower;

Her = ((mcinv(Mel))./uT).*nhRow; % convert from Mel scale units to Hertz scale down range to 0 - 220
filterbank = createFilterBank(Her');

output = zeros(plots,nCol);
for i = 1:nCol
    for j = 1:plots
        output(j,i) = log(dot(filterbank(:,j), P(:,i))); % take sigma of product of power spectrum and Mel Filterbank, and then log
    end
    output(:,i) = dct(output(:,i)); % compute DCT
end

% take first 15 coefficients
FINAL = output(1:15,:);
end
