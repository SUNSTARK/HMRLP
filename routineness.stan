data{
    int<lower=1> N;	// number of users
    int<lower=1> M;	// number of data points
    int<lower=0> W;	// range of week
    int<lower=0> J;	// range of day-hour
    int<lower=0> y[M];	// outcome of observation m (drive times)
}
parameters{
    vector[W] alpha[N];
    vector[J] mu;
    vector[W] gamma[N];
    vector[J-1] eta_free[N];  // in order to fix eta_i0 = 0
    
    // hyper-parameters in GPs
    real<lower=0> tau;  // alpha's variance, should be positive
    real gamma_0;  // gamma's mean, shouldn't be declared positive
    real<lower=0> sigma_gamma;
    real<lower=0> sigma_mu;
    real<lower=0> rho_mu;
    cholesky_factor_corr[7] L_omega_routine;
    cholesky_factor_corr[7] L_omega_random;
    real<lower=0> sigma_eta;
    real<lower=0> rho_eta;
}
transformed parameters{
    vector[J] eta[N];
    corr_matrix[7] omega_routine;  // a correlation matrix over days of the week
    corr_matrix[7] omega_random;  // a correlation matrix over days of the week

    // mean function in GPs
    vector[J] mu_mean = rep_vector(0, J);
    vector[J] eta_mean = rep_vector(0, J);
    vector[W] gamma_mean = rep_vector(gamma_0, W);

    omega_routine = multiply_lower_tri_self_transpose(L_omega_routine);
    omega_random = multiply_lower_tri_self_transpose(L_omega_random);

   // make everyone's eta_0 = 0
    for(n in 1:N) {
        eta[n] = append_row(0, eta_free[n]);
    }
}
model{
    matrix[J,J] phi_mu;
    matrix[J,J] phi_mu1;
    matrix[W,W] phi_gamma;
    matrix[W,W] phi_gamma1;
    matrix[J,J] phi_eta;
    matrix[J,J] phi_eta1;
  
    // some week priors
    sigma_gamma ~ normal(0,5);
    sigma_mu ~ normal(0,5);
    sigma_eta ~ normal(0,5);
    rho_mu ~ normal(0,10);
    rho_eta ~ normal(0,10);
    L_omega_routine ~ lkj_corr_cholesky(2);
    L_omega_random ~ lkj_corr_cholesky(2);
    tau ~ normal(0,5);
    gamma_0 ~ normal(0,5);
  
    // mu, random rate, public
    for(i in 1:J) {
        int d = i %/% 24;
        int h = i % 24;
        if(h != 0) {
            d += 1;
        }
        for(j in i:J) {
            int d2 = j %/% 24;
            int h2 = j % 24;
            if(h2 != 0) {
                d2 += 1;
            }
            phi_mu[i,j] = sigma_mu * omega_random[d,d2] * exp(-2*(sin(pi()*abs(h-h2)/24)^2)/rho_mu);
            phi_mu[j,i] = phi_mu[i,j];
        }
        phi_mu[i,i] = phi_mu[i,i]+0.0001;
    }
    phi_mu1 = cholesky_decompose(phi_mu);
    mu ~ multi_normal_cholesky(mu_mean, phi_mu1);

    // eta, routine rate, private
    for(i in 1:J) {
        int d = i %/% 24;
        int h = i % 24;
        if(h != 0) {
            d += 1;
        }
        for(j in i:J) {
            int d2 = j %/% 24;
            int h2 = j % 24;
            if(h2 != 0) {
                d2 += 1;
            }
            phi_eta[i,j] = sigma_eta * omega_routine[d,d2] * exp(-2*(sin(pi()*abs(h-h2)/24)^2)/rho_eta);
            phi_eta[j,i] = phi_eta[i,j];
        }
        phi_eta[i,i] = phi_eta[i,i]+0.0001;
    }
    phi_eta1 = cholesky_decompose(phi_eta);

    // gamma, routine scaling term, private
    for(i in 1:W) {
        for(j in i:W) {
            phi_gamma[i,j] = sigma_gamma * exp(-0.5*((i-j)^2)/9);
            phi_gamma[j,i] = phi_gamma[i,j];
        }
        phi_gamma[i,i] = phi_gamma[i,i]+0.0001;
    }
    phi_gamma1 = cholesky_decompose(phi_gamma);
  
    for(n in 1:N){
        vector[W*J] lambda;
        vector[W*J] theta;
    
        // alpha, random scaling term, private
        alpha[n,2:W] ~ normal(alpha[n,1:(W-1)], tau);
        gamma[n] ~ multi_normal_cholesky(gamma_mean, phi_gamma1);
        eta[n] ~ multi_normal_cholesky(eta_mean, phi_eta1);
    
        // likelihood
        for(w in 1:W) {
            for(j in 1:J) {
                int time_i = (w-1)*J+j;
                int data_i = time_i+(n-1)*W*J;
                lambda[time_i] = exp(alpha[n,w]+mu[j]) + exp(gamma[n,w]+eta[n,j]);
                theta[time_i] = -lambda[time_i];
                if(y[data_i] != 0) {
                    theta[time_i] += y[data_i]*log(lambda[time_i]);
                }
            }
        }
        target += theta;
    }
}