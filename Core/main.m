%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             Author: Rohan Karnawat                    %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ditermensiterons : 15 * 750 * 400
nRow = 1;
nCol = 135;

dataset = zeros(nRow*400,nCol); % Fiternal dataset

% easily add genres, increment number of rows by 15 * 100
genre = dir('pop/*.au');
for iter = 1:length(genre)
    dataset(iter,:) = mfcc('pop', genre(iter).name);
end

genre = dir('jazz/*.au');
for iter = 1:length(genre)
    dataset(101 + iter,:) = mfcc('jazz',genre(iter).name);
end

genre = dir('metal/*.au');
for iter = 1:length(genre)
    dataset(201 + iter,:) = mfcc('metal',genre(iter).name);
end

genre = dir('classical/*.au');
for iter = 1:length(genre)
    dataset(301 + iter,:) = mfcc('classical', genre(iter).name);
end
