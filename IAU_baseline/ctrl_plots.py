import matplotlib.pyplot as plt

# plotting residual and 2-sigma range
def plot_residual(time,res,sd,flag):
    sdp_array = [2*sd] * len(res)
    sdm_array = [-2*sd] * len(res)
 
    plt.figure(figsize=(10,8))                 
    plt.scatter(time,res,c=flag)
    plt.plot(time,sdp_array,color='orange',linestyle='--')
    plt.plot(time,sdm_array,color='orange',linestyle='--')

    
# plotting data with fit result and 2-sigma range
def plot_timeseries(time, mxr, yfit, sd, flag):
    sdp_array = [None] * len(yfit)
    sdp_array = yfit + 2*sd
 
    sdm_array = [None] * len(yfit)
    sdm_array = yfit - 2*sd
    
    plt.figure(figsize=(10,8))                 
    plt.scatter(time,mxr,c=flag)
    plt.plot(time,yfit,color='red')
    plt.plot(time,sdp_array,color='orange',linestyle='--')
    plt.plot(time,sdm_array,color='orange',linestyle='--')
    