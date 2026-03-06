# Linear Regression

Linear Regression is supervised learning algorithm. Meaning it learns from labeled data.

## Explaintion

Training data :

| House size (100 sqft) | House price (lakh) |
| --------------------- | ------------------ |
| 1                     | 2                  |
| 2                     | 4                  |
| 3                     | 6                  |

So, algorithm aims to find the linear equation which can predict house price for new houses (new unseen data). As we know linear equation can be represented using slope intercept form i.e `y = mx + b`

Now, if we plot the graph of above data, we will have house size on X-axis & house price on Y-axis.

So, in `y = mx + b`,
y is any point on Y-axis  
x is any point on X-axis  
m is slope (how much y changes when x increases by 1 unit)  
b is y-intercept ( point on y-axis where straight line (formed using linear equation) cross/intercept Y-axis)

So now we have equation `y = mx + b` where y will be house price and x will be that house's size.

_So algorithm need to find optimal value of m and b so that it outputs correct value of y (house price) for its corresponding x (house size)_

Example, we take m = 0, b = 0  
let's select 1st data point
x = 1 (100 sqft)
y = 2 (lakh)  
So, data point is (x,y) = (1,2)

Put all values in equation,

y = mx + b  
2 = 1(0) + 0  
2 != 0

So when we take m and b as 0 for x = 1 (i.e house size 1 (100 sqft)), equation gives y = 0 (i.e 0 lakh), which is incorrect prediction. Because Actual value of y (house price) is 2 (in lakh) when x (house size) is 1 (100 sqft)
