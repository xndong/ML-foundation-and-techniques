Polynomial nonlinear transform(a.k.a.多项式非线性变换,通过多项式进行nonlinear transform)：
1. a uniary polynomial regression(a.k.a. 一元多项式回归),order = 3.   x->1, x, x^2, x^3   
2. a multi polynomial regression(a.k.a. 多元多项式回归),order = 2.    (x1,x2,x3,x4...xn)->(1,x1,x2,x3,.....xn,
																							x1^2,x2^2.....xn^2,
                                                                                         	x1x2,x1x3,....x1xn,
                                                                             			 	x2x3,x2x4,....x2xn,
                                                                             			 	x3x4,x3x5,....x3xn,
                                                                             			 	.....
                                                                             			 	xn-1xn)    

在大多数资料里通常把1叫做多项式回归，把2叫做nonlinear regression。英文搜索polynomial regression查出来的也是按这个约定叫的，所以我们实现的时候就按照这个约定俗称的叫法来。
但是广义上，二者都是多项式回归。