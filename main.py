# Dexter Dysthe
# Dr. Zheng
# B9324
# 8 October 2021

import statsmodels.api as sm
import pandas as pd
import numpy as np

from patsy import dmatrices


nls = pd.read_csv('nls_2008.txt', sep="\t", names=['luwe', 'educ', 'exper', 'age', 'fed', 'med', 'kww', 'iq', 'white'])


# ------------------------------------------------- Question 1 ------------------------------------------------- #
print('\n')
print('------------------------------------------------ Question 1 ------------------------------------------------')
print('\n')

# Summary statistics of each of the columns from the nls dataset
results_dict = {'Minimum': nls.min(), 'Maximum': nls.max(), 'Mean': nls.mean(), 'SD': nls.std(ddof=1)}
results = pd.DataFrame(results_dict, index=['luwe', 'educ', 'exper', 'age', 'fed', 'med', 'kww', 'iq', 'white'])
print(results, '\n')
print('\n')


# ------------------------------------------------- Question 2 ------------------------------------------------- #
print('\n')
print('------------------------------------------------ Question 2 ------------------------------------------------')
print('\n')

# Create experience squared column
nls['exper_squared'] = nls['exper'] ** 2

# (1) Create list of variables for regression and (2) restrict the nls data frame just to these variables
vars_of_interest = ['luwe', 'educ', 'exper', 'exper_squared']
nls_for_ols = nls[vars_of_interest]

y, X = dmatrices('luwe ~ educ + exper + exper_squared', data=nls_for_ols, return_type='dataframe')
nls_model = sm.OLS(y, X)

# Conventional standard errors and HET-robust standard errors
nls_res = nls_model.fit()
nls_res_robust = nls_model.fit(cov_type='HC0')

print('\n')
print('---------------------------------------- Conventional Standard Errors ----------------------------------------')
print('\n')
print(nls_res.summary(), '\n')

print('\n')
print('------------------------------------------- Robust Standard Errors -------------------------------------------')
print('\n')
print(nls_res_robust.summary(), '\n')


# Displays the non-robust and robust covariance matrix for Betas
print('\n')
print('-------------------- Covariance Matrix of Betas --------------------')
print('--------------------------------------------------------------------')
np.set_printoptions(suppress=True)
print("Conventional: \n", nls_res.cov_params(), '\n')
print("HET-Robust: \n", nls_res_robust.cov_params(), '\n')

# Displays the estimated variance of the residual
print('\n')
print('------------------------ Variance of Residuals ------------------------')
print('-----------------------------------------------------------------------')
np.set_printoptions(suppress=True)
sigma_hat_squared = sum(nls_res.resid.values ** 2) / nls_res.df_resid
print("Estimated Variance of Residual:", sigma_hat_squared, '\n')

# Displays the non-robust and robust covariance matrix for all regression parameters.
# The last diagonal entry is twice the square of the estimator for the residual
# variance -- that is, the asymptotic variance of the estimator sigma squared hat --
# as given in Lecture Notes 3.
print('\n')
print('---------------- Covariance Matrix of All Parameters (Omega hat) ----------------')
print('---------------------------------------------------------------------------------')
all_params = nls_res.cov_params()
all_params['Resid Var'] = [0, 0, 0, 0]
all_params.loc['Resid Var'] = [0, 0, 0, 0, 2*(sigma_hat_squared**2)]

all_params_robust = nls_res_robust.cov_params()
all_params_robust['Resid Var'] = [0, 0, 0, 0]
all_params_robust.loc['Resid Var'] = [0, 0, 0, 0, 2*(sigma_hat_squared**2)]

np.set_printoptions(suppress=True)
print("Conventional: \n", all_params, '\n')
print("HET-Robust: \n", all_params_robust, '\n')


# ------------------------------------------------- Question 3 ------------------------------------------------- #
# Fully answered in the hand-written solutions.

print('\n')
print('------------------------------------------------ Question 3 ------------------------------------------------')
print('\n')

coefficient_beta_3 = (2 * nls_for_ols['exper'] + 1).mean()
print("Mean experience", nls_for_ols['exper'].mean())
print("Predicted effect on average log earnings: ", -nls_res.params[1] + nls_res.params[2] +
      coefficient_beta_3 * nls_res.params[3])


# ------------------------------------------------- Question 4 ------------------------------------------------- #
# Supplemental answer in the above hand-written solutions.

print('\n')
print('------------------------------------------------ Question 4 ------------------------------------------------')

# Implementing the regression detailed in the above hand-written solutions.
nls['educ_exp'] = nls['educ'] + nls['exper']
nls['exp_squared_adj'] = (((nls['exper'] + 1) ** 2).mean() - nls['exper_squared'].mean() - nls['exper']) * (-1) * nls['exper']

vars_of_interest2 = ['luwe', 'educ_exp', 'exper', 'exp_squared_adj']
nls_for_ols2 = nls[vars_of_interest2]

y_2, X_2 = dmatrices('luwe ~ educ_exp + exper + exp_squared_adj', data=nls_for_ols2, return_type='dataframe')
nls_model2 = sm.OLS(y_2, X_2)
nls_res2 = nls_model2.fit()

print('\n')
print(nls_res2.summary(), '\n')
print("Estimate of Theta: ", nls_res2.params[2])


# ------------------------------------------------- Question 5 ------------------------------------------------- #
# Supplemental answer in the above hand-written solutions.

print('\n')
print('------------------------------------------------ Question 5 ------------------------------------------------')
print('\n')

# Store the fitted values
beta_hat_0 = nls_res.params.values[0]
beta_hat_1 = nls_res.params.values[1]
beta_hat_2 = nls_res.params.values[2]
beta_hat_3 = nls_res.params.values[3]

# Create new dataframe containing only those individuals with less than 12 years of education
nls_12_or_less = nls[nls['educ'] < 12]

# Created to simplify the below expressions used to calculate theta_hat
adjusted_exper = nls_12_or_less['exper'] - (12 - nls_12_or_less['educ'])
educ = nls_12_or_less['educ']
exper = nls_12_or_less['exper']

# Create vectors for the values to be exponentiated
vec_for_first_sum = (beta_hat_0 + 12*beta_hat_1 + adjusted_exper*beta_hat_2 + (adjusted_exper**2)*beta_hat_3 \
                    + 0.5*sigma_hat_squared).to_numpy()
vec_for_second_sum = (beta_hat_0 + educ*beta_hat_1 + exper*beta_hat_2 + (exper**2)*beta_hat_3 \
                    + 0.5*sigma_hat_squared).to_numpy()

# Use the expression for theta_hat shown in the above hand-written solutions.
theta_hat = 1/930 * (np.sum(np.exp(vec_for_first_sum) - np.exp(vec_for_second_sum)))
print("Theta hat: ", theta_hat)

# ------------------------------------------------- Question 6 ------------------------------------------------- #
# Supplemental answer in the above hand-written solutions.

print('\n')
print('------------------------------------------------ Question 6 ------------------------------------------------')
print('\n')

N = 930

# Compute the estimate of the gradient vector of the function g(gamma) shown in the above hand-written solutions.
# Compute the estimate of the gradient vector of the function g(gamma) shown in the above hand-written solutions.
partial_g_1_hat = 1/930 * (np.sum(12*np.exp(vec_for_first_sum) - educ*np.exp(vec_for_second_sum)))
partial_g_2_hat = 1/930 * (np.sum(adjusted_exper*np.exp(vec_for_first_sum) - exper*np.exp(vec_for_second_sum)))
partial_g_3_hat = 1/930 * (np.sum((adjusted_exper**2)*np.exp(vec_for_first_sum) - (exper**2)*np.exp(vec_for_second_sum)))

# The partial derivative of g with respect to beta_0 equals g itself, and the partial derivative of g with respect
# to sigma squared is 1/2 of g itself. Since we need to estimate gamma, these reduce to theta_hat and 1/2*theta_hat
# respectively.
grad_g_hat = np.array([theta_hat, partial_g_1_hat, partial_g_2_hat, partial_g_3_hat, 0.5*theta_hat])
print("Gradient of g: ")
print(grad_g_hat, '\n')


# ------------------------------------ Conventional Standard Errors ------------------------------------ #

np.set_printoptions(suppress=True)
print('--------------------- Conventional Standard Errors --------------------- \n')

# The cov_params() function returns the covariance matrix of Beta_hat. Since we want the covariance matix of
# sqrt(N)*Beta_hat, we must adjust the output of nls_res.cov_params().
v_hat = N * nls_res.cov_params().to_numpy()

# Add the row and column for the residual variance
omega_hat = np.vstack((np.column_stack((v_hat, np.array([0, 0, 0, 0]))), np.array([0, 0, 0, 0, 2*(sigma_hat_squared**2)])))

print("Omega hat matrix: \n")
print(omega_hat)
print('\n')
print("Standard Error: ", np.sqrt(grad_g_hat.dot(omega_hat.dot(grad_g_hat))/N), "\n")

# -------------------------------------- HET-Robust Standard Errors ------------------------------------- #

np.set_printoptions(suppress=True)
print('----------------------- Robust Standard Errors ------------------------- \n')

v_hat_robust = N * nls_res_robust.cov_params().to_numpy()
omega_hat_robust = np.vstack((np.column_stack((v_hat_robust, np.array([0, 0, 0, 0]))), np.array([0, 0, 0, 0, 2*(sigma_hat_squared**2)])))

print("Omega hat: \n")
print(omega_hat)
print('\n')
print("Standard Error: ", np.sqrt(grad_g_hat.dot(omega_hat_robust.dot(grad_g_hat))/N), "\n")


# ------------------------------------------------- Question 7 ------------------------------------------------- #
print('\n')
print('------------------------------------------------ Question 7 ------------------------------------------------')
print('\n')

# Create a copy of the nls_for_ols dataframe as I am going to rename the column.
nls_bootstrap = nls_for_ols.copy(deep=True)
nls_bootstrap.rename(columns={'luwe': 'luwe_tilde'}, inplace=True)


def theta_hat(dataframe, beta_0, beta_1, beta_2, beta_3, sigma_squared):
    """
    The purpose of this function is to assist with the below bootstrap calculations for
    both Q7 and Q8. Each iteration of the bootstrap produces a new dataframe on which we
    wish to run OLS in order to obtain values for beta hat and sigma squared hat. Thus,
    the purpose of this function is to calculate the predicted effect of the policy given
    the bootstrapped dataframe and the bootstrapped beta hat and sigma squared hat values.
    """
    # Create new dataframe containing only those individuals with less than 12 years of education
    nls_12_or_less_boot = dataframe[dataframe['educ'] < 12]

    # Created to simplify the below expressions used to calculate theta_hat
    adjusted_exper_boot = nls_12_or_less_boot['exper'] - (12 - nls_12_or_less_boot['educ'])
    educ_boot = nls_12_or_less_boot['educ']
    exper_boot = nls_12_or_less_boot['exper']

    # Create vectors for the values to be exponentiated
    vec_for_first_sum_boot = (beta_0 + 12*beta_1 + adjusted_exper*beta_2 + (adjusted_exper_boot**2)*beta_3 \
                    + 0.5*sigma_squared).to_numpy()
    vec_for_second_sum_boot = (beta_0 + educ*beta_1 + exper*beta_2 + (exper**2)*beta_3 \
                    + 0.5*sigma_squared).to_numpy()

    # Use the expression for theta_hat shown in the above hand-written solutions.
    theta_hat_boot = 1/930 * (np.sum(np.exp(vec_for_first_sum_boot) - np.exp(vec_for_second_sum_boot)))

    return theta_hat_boot


# This list will have 1d numpy arrays as entries, each numpy array corresponding to the vector of
# estimated beta values for that iteration of the bootstrap
bootstrap_theta = []

for b in range(100000):
    # Draw a sample, with replacement, of size 930 from the collection of our regression residuals
    bootstrap = np.array(nls_res.resid.sample(n=930, replace=True))

    # Construct the bth sample values for log earnings and replace the original luwe entries with these
    # new values.
    sample_values = beta_hat_0 + beta_hat_1 * nls_bootstrap['educ'] + beta_hat_2 * nls_bootstrap['exper'] + \
                    beta_hat_3 * nls_bootstrap['exper_squared'] + bootstrap
    nls_bootstrap['luwe_tilde'] = sample_values

    # Run the regression using the bootstrapped data
    y_b, X_b = dmatrices('luwe_tilde ~ educ + exper + exper_squared', data=nls_bootstrap, return_type='dataframe')
    bootstrap_model_res = sm.OLS(y_b, X_b).fit()

    # Obtain vector of estimated beta values as well as the estimated sigma squared value, and apply the
    # the theta_hat(...) function defined above to these arguments
    beta_boot = bootstrap_model_res.params.to_numpy()
    sigma_squared_boot = sum(bootstrap_model_res.resid.values ** 2) / bootstrap_model_res.df_resid
    theta = theta_hat(nls_bootstrap, beta_boot[0], beta_boot[1], beta_boot[2], beta_boot[3], sigma_squared_boot)
    bootstrap_theta.append(theta)

# Convert the list bootstrap_theta to a numpy array and then calculate the standard error
np_bootstrap_theta = np.array(bootstrap_theta)
print("Standard Error of Policy: ", np.std(np_bootstrap_theta))


# ------------------------------------------------- Question 8 ------------------------------------------------- #
print('\n')
print('------------------------------------------------ Question 8 ------------------------------------------------')
print('\n')

# This list will have 1d numpy arrays as entries, each numpy array corresponding to the vector of
# estimated beta values for that iteration of the bootstrap
bootstrap_theta2 = []

for b in range(100000):
    # Draw a sample, with replacement, of size 930 from the rows of the dataframe nls_for_ols
    bootstrap2 = nls_for_ols.sample(n=930, replace=True)

    # Run the regression using the bootstrapped data
    y_b2, X_b2 = dmatrices('luwe ~ educ + exper + exper_squared', data=bootstrap2, return_type='dataframe')
    bootstrap_model_res2 = sm.OLS(y_b2, X_b2).fit()

    # Obtain vector of estimated beta values as well as the estimated sigma squared value, and apply the
    # the theta_hat(...) function defined above to these arguments
    beta_boot2 = bootstrap_model_res2.params.to_numpy()
    sigma_squared_boot2 = sum(bootstrap_model_res2.resid.values ** 2) / bootstrap_model_res2.df_resid
    theta_boot2 = theta_hat(nls_bootstrap, beta_boot2[0], beta_boot2[1], beta_boot2[2], beta_boot2[3], sigma_squared_boot2)
    bootstrap_theta2.append(theta_boot2)

# Convert the list bootstrap_theta2 to a numpy array and then calculate the standard error
np_bootstrap_theta2 = np.array(bootstrap_theta2)
print("Standard Error of Policy: ", np.std(np_bootstrap_theta2))
