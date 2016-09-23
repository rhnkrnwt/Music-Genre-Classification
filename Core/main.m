%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             Author: Rohan Karnawat                    %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ditermensiterons : 15 * 750 * 400
nRow = 15;
nCol = 750;

dataset = zeros(nRow*400,nCol); % Fiternal dataset

% easily add genres, increment number of rows by 15 * 100
genre = dir('pop/*.au'); 
for iter = 1:length(genre)
    startIndex = 1+(iter-1)*nRow;
    endIndex = startIndex + nRow-1;
    dataset(startIndex:endIndex,:) = mfcc('pop', genre(iter).name);
end

genre = dir('jazz/*.au');
for iter = 1:length(genre)
    startIndex = 1501+(iter-1)*nRow;
    endIndex = startIndex + nRow-1;
    dataset(startIndex:endIndex,:) = mfcc('jazz',genre(iter).name);
end

genre = dir('metal/*.au');
for iter = 1:length(genre)
    startIndex = 3001+(iter-1)*nRow;
    endIndex = startIndex + nRow-1;
    dataset(startIndex:endIndex,:) = mfcc('metal',genre(iter).name);
end

genre = dir('classical/*.au');
for iter = 1:length(genre)
    startIndex = 4501+(iter-1)*nRow;
    endIndex = startIndex + nRow-1;
    dataset(startIndex:endIndex,:) = mfcc('classical', genre(iter).name);
end
