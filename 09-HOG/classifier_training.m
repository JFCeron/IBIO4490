
%Trains a support vector machine in order to recognize faces
%Hyperparameters :
%conf.svm.solver choose between sgd or sdca
%TODO añadir este último hiperparametro al main
%conf.modelPath = fullfile(conf.dataDir, '-model.mat') ;
%C del SVM
% conf.svm.C
%Controla la distancia del plano a un respectivo margen
%conf.svm.biasMultiplier = 1 ;

%Parametros: 
%positiveFeatures, el return del metodo get_positivefeatures
%negativeFeatures, el return del metodo get_negative features

function[w,b] = classifier_training(positiveFeatures,negativeFeatures,conf)

  dimensionesPositivas = size(positiveFeatures);
  dimensionesNegativas = size(negativeFeatures);

  LabelsPositivos = repmat( 1 , [1,dimensionesPositivas(1)]);
  LabelsNegativos = repmat( -1 , [1,dimensionesNegativas(1)]);
  
  MatrizEntrenamiento = [transpose(positiveFeatures),transpose(negativeFeatures)];  
  vectorLabels = [LabelsPositivos,LabelsNegativos];
      	tamanioMuestra = size(vectorLabels) ;
        lambda =  1/(conf.svm.C*tamanioMuestra(2)) ;
          fprintf('Training model') ;
          y = vectorLabels ;
          [w b info] = vl_svmtrain(MatrizEntrenamiento, vectorLabels, lambda, ...
            'Solver', conf.svm.solver, ...
            'MaxNumIterations', 50/lambda, ...
            'BiasMultiplier', conf.svm.biasMultiplier, ...
            'Epsilon', 1e-3);

    model.b = conf.svm.biasMultiplier * b ;
    model.w = w ;

    
