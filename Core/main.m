%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             Author: Rohan Karnawat                    %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ditermensiterons : 15 * 750 * 400
nRow = 1;
nCol = 135;

dataset = zeros(nRow*500,nCol); % Fiternal dataset

% easily add genres, increment number of rows by 15 * 100
genre = dir('pop/*.au');
for iter = 1:length(genre)
    dataset(iter,:) = mfcc('pop', genre(iter).name);
end

genre = dir('jazz/*.au');
for iter = 1:length(genre)
    dataset(100 + iter,:) = mfcc('jazz',genre(iter).name);
end

genre = dir('metal/*.au');
for iter = 1:length(genre)
    dataset(200 + iter,:) = mfcc('metal',genre(iter).name);
end

genre = dir('classical/*.au');
for iter = 1:length(genre)
    dataset(300 + iter,:) = mfcc('classical', genre(iter).name);
end

genre = dir('hiphop/*.au');
for iter = 1:length(genre)
    dataset(400 + iter,:) = mfcc('hiphop', genre(iter).name);
end


principal_data = red_dim(dataset);
