**Linear Regression**

Linear regression is used to estimate real world values like cost of houses, number of calls, total sales etc. based on continuous variable(s). Here, we establish relationship between dependent and independent variables by fitting a best line. This line of best fit is known as regression line and is represented by the linear equation Y= a * X + b.

or for the purposes of this class ***y = w0 + w1 * x***
![](/Homework1/images/LRgraphExample.JPG?raw=true)

In this equation:

Y – Dependent Variable

a – Slope

X – Independent variable

b – Intercept

These coefficients a and b are derived based on minimizing the sum of squared difference of distance between data points and regression line.



**Example:**

The best way to understand linear regression is by considering an example. Suppose we are asked to arrange students in a class in the increasing order of their weights. By looking at the students and visually analyzing their heights and builds we can arrange them as required using a combination of these parameters, namely height and build. This is real world linear regression example. We have figured out that height and build have correlation to the weight by a relationship, which looks similar to the equation above.

**Types of Linear Regression**

Linear Regression is of mainly two types: Simple Linear Regression and Multiple Linear Regression. Simple Linear Regression is characterized by one independent variable while Multiple Linear Regression is characterized by more than one independent variables. While finding the line of best fit, you can fit a polynomial or curvilinear regression. You can use the following code for this purpose.

    import matplotlib.pyplot as plt
    plt.scatter(X, Y)
    yfit = [a + b * xi for xi in X]
    plt.plot(X, yfit)

**Building a Linear Regressor**

Regression is the ***process of estimating the relationship between input data and the continuous-valued output data.*** This data is usually in the form of real numbers, and our goal is to estimate the underlying function that governs the mapping from the input to the output.

Consider a mapping between input and output as shown:

1 --> 2

3 --> 6

4.3 --> 8.6

7.1 --> 14.2

You can easily estimate the relationship between the inputs and the outputs by **analyzing the pattern**. We can observe that the output is twice the input value in each case, hence the **transformation would be: f(x) = 2x**
Linear regression refers to estimating the relevant function using a linear combination of input variables. The preceding example was an example that consisted of one input variable and one output variable.

***The goal of linear regression is to extract the relevant linear model that relates the input variable to the output variable.*** This aims to minimize the sum of squares of differences between the actual output and the predicted output using a linear function. This method is called Ordinary Least Squares. You may assume that a curvy line out there that fits these points better, but linear regression does not allow this. The main advantage of linear regression is that it is not complex. You may also find more accurate models in non-linear regression, but they will be slower. Here the model tries to approximate the input data points using a straight line.

***Let us understand how to build a linear regression model in Python.***

Consider that you have been provided with a data file, called data_singlevar.txt. This contains comma-separated lines where the first element is the input value and the second element is the output value that corresponds to this input value. You should use this as the input argument:

Assuming line of best fit for a set of points is:

    y = a + b * x
    where: b = ( sum(xi * yi) - n * xbar * ybar ) / sum((xi - xbar)^2)
    a = ybar - b * xbar

    Use the following code for this purpose:
    # sample points
    X = [0, 6, 11, 14, 22]
    Y = [1, 7, 12, 15, 21]

    # solve for a and b
    def best_fit(X, Y):
        xbar = sum(X)/len(X)
        ybar = sum(Y)/len(Y)
        n = len(X) # or len(Y)
        numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
        denum = sum([xi**2 for xi in X]) - n * xbar**2
        b = numer / denum
        a = ybar - b * xbar
        print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
        return a, b

    # solution
    a, b = best_fit(X, Y)
    #best fit line:
    #y = 0.80 + 0.92x
    # plot points and fit line
    import matplotlib.pyplot as plt
    plt.scatter(X, Y)
    yfit = [a + b * xi for xi in X]
    plt.plot(X, yfit)
    plt.show()
    best fit line:
    y = 1.48 + 0.92x

If you run the above code, you can observe the output graph as shown:

![Linear Regression Graph](/Homework1/images/LinearRegressionGraph.jpg?raw=true)

***Note*** that this example uses only the first feature of the diabetes dataset, in order to illustrate a **two-dimensional plot** of this regression technique. The straight line can be seen in the plot, ***showing how linear regression attempts to draw a straight line that will best minimize the residual sum of squares between the observed responses in the dataset***, and the responses predicted by the linear approximation.

You can **calculate the coefficients**, the residual **sum of squares** and the **variance score** using the program code shown below:

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-30]
    diabetes_X_test = diabetes_X[-30:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-30]
    diabetes_y_test = diabetes.target[-30:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)

    # The mean squared error
    print("Mean squared error: %.2f"
        % mean_squared_error(diabetes_y_test, diabetes_y_pred))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

***You can observe the following output once you execute the code given above:***

    Automatically created module for IPython interactive environment
    ('Coefficients: \n', array([ 941.43097333]))
    Mean squared error: 3035.06
    Variance score: 0.41
