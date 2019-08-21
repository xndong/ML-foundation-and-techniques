Increase the dimension of input features by nonlinear transform. Thus, model can be more complexity.
Polynomial nonlinear tranform is a popularly used nonlinear transform. I also think there are other kinds of nonlinear transform, but they will not be considered here. You can add any nonlinear transform, e.g.(x1,x2,x3)-> (sin(x1),sin(x2),sin(x3)...,sin(xn),x1^2,x2^2...xn^2)

Note that overfit can emerge when using nonlinear transform. So, regularization is necessary. Lasso, Ridge and ElasticNet are popularly used regularization items.
Lasso Regression
Ridge Regression
ElasticNet Regression