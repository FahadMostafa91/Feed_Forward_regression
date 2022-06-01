%Feed-Forward Regression Network (Commands)
%Instructions are in the task pane to the left. Complete and submit each task one at a time.

%This code loads fuel economy data.
load fuel
whos econ carData

%Tasks 1, 3, & 4
%TASK
%Initialize a neural network named net with one hidden layer containing 15 neurons.
%Train the network with the data in carData and use the target values in econ. Save the training record in a structure named tr.
%Note that the samples for the data should be along the matrix columns rather than rows as input to the train function.
%Initialize and train the neural network
net = fitnet(15);
carData = carData';
econ = econ';
[net,tr] = train(net,carData,econ);

%Task 2
%Make predictions named econPred on the test set.
%Then, you may use plotregression to show how well the predictions do.
%Predict response and evaluate network performance

econPred = net(carData(:,tr.testInd));
plotregression(econ(tr.testInd),econPred)