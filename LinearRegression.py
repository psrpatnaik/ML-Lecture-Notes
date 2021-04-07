# y = weight * x  + bias
x = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
y = [64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211]


# MSE
def cost_function(x, y, weight, bias):
    length = len(x)
    total_error = 0.0
    for i in range(length):
        total_error += (y[i] - (weight*x[i] + bias))**2
    return total_error / length


def update_weights(x, y, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    length = len(x)

    for i in range(length):
        # Calculate partial derivatives
        # -2x(y - ( weight x + bias))
        weight_deriv += -2*x[i] * (y[i] - (weight*x[i] + bias))

        # -2(y - (weight x + bias))
        bias_deriv += -2*(y[i] - (weight*x[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / length) * learning_rate
    bias -= (bias_deriv / length) * learning_rate
    return weight, bias

def train(x, y, weight, bias, learning_rate):
    cost_history = []
    i=1
    while True:
        weight,bias = update_weights(x, y, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(x, y, weight, bias)
        cost_history.append(cost)

        if cost < 0.0001:
            break
        if len(cost_history) > 1:
            if cost_history[-1]>cost_history[-2]:
                break
        # Log Progress

        print ("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))
        i+=1
    return weight, bias, cost_history

train(x,y,1,1,0.0001)