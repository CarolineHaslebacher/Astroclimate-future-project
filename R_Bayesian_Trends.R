# run with Rscript /home/haslebacher/chaldene/Astroclimate_Project/sites/R_Bayesian_Trends.R
# You must select manually for which variable you want to calculate the Bayesian trends
# (lines 19-25)

library(rethinking)
library(ggplot2)
library("rstan")

options(mc.cores = parallel::detectCores())
# To avoid recompilation of unchanged Stan programs, we recommend calling
rstan_options(auto_write = TRUE)
# 

# load in functions written in R_Astroclim.R
# getwd() should be set to '/home/haslebacher' or '/home/caroline'
source('./chaldene/Astroclimate_Project/sites/R_Astroclim.R')
######### choose a variable

# variable <- 'total_cloud_cover' # test! (and multiply by 100 to get percent!!); calc on 2020-11-23
# variable <- 'T' # calc
# variable <- 'SH' # test (and convert to g/kg); calc on 2020-11-23
# variable <- 'RH' # calc on 2020-11-23
# variable <- 'TCW' # test
variable <- 'seeing_osborn' # calc on 2020-11-19
# variable <- 'wind_speed_seeing' # calc on 2020-11-19

##########################################


if (variable == 'total_cloud_cover'){
  print('cloud_cover')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/total_cloud_cover/future_trends/"
  ylabel <- 'fraction of clouds [%]'
  unit <- '%'
  offset <- 30
  cpressure <- c('_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level') # '_925' eg
  
  
} else if (variable == 'T'){
  print('temperature')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/T/future_trends/"
  ylabel <- 'Temperature [deg C]'
  unit <- 'deg C'
  offset <- 9
  # cpressure <- c('_single_level', '_single_level', '_850', '_single_level','_850', '_925', '_single_level', '_700') # '_925' eg
  # new variables from new workflow (best matches based on comparison of PRIMAVERA to ERA5 (and not to in-situ))
  # cpressure <- c('_600', '_700', '_850', '_850','_700', '_single_level', '_single_level', '_700') # '_925' eg
  # workflow 4.1:
  cpressure <- c('_600', '_700', '_850', '_850','_850', '_925', '_850', '_700')
  
} else if (variable == 'SH'){
  print('specific humidity')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/SH/future_trends/"
  ylabel <- 'Specific Humidity [g/kg]'
  # cpressure <- c('_700', '_850', '_850', '_850','_850', '_925', '_850', '_700') # '_925' eg
  # new variables from new workflow (best matches based on comparison of PRIMAVERA to ERA5 (and not to in-situ))
  # cpressure <- c('_600', '_850', '_850', '_850','_850', '_925', '_850', '_700') # '_925' eg
  # workflow 4.1:
  cpressure <- c('_600', '_700', '_850', '_850','_850', '_925', '_850', '_700')
  unit <- 'g/kg'
  offset <- 2
  
} else if (variable == 'RH'){
  print('relative humidity')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/RH/future_trends/"
  ylabel <- 'Relative Humidity [%]'
  # cpressure <- c('_600', '_700', '_850', '_850','_850', '_1000', '_850', '_700')
  # new variables from new workflow (best matches based on comparison of PRIMAVERA to ERA5 (and not to in-situ))
  # cpressure <- c('_600', '_850', '_850', '_850','_850', '_925', '_850', '_700') # paranal from 700 to 850 hPa, also: take siding spring 925hPa (too many nan otherwise)
  # workflow 4.1:
  cpressure <- c('_600', '_700', '_850', '_850','_850', '_925', '_850', '_700')
  unit <- '%'
  offset <- 10
  
} else if (variable == 'seeing_osborn') {
  print('seeing (osborn)')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/seeing_nc/future_trends/osborn/"
  ylabel <- 'Astronomical seeing [arcesc]'
  # cpressure <- c('_200', '_200', '_200', '_200','_200', '_200', '_200', '_200')
  cpressure <- c('_5', '_5', '_5', '_5','_5', '_5', '_5', '_5')
  unit <- 'arcsec'
  offset <- 0.6
  
} else if (variable == 'wind_speed_seeing'){
  print('seeing (wind speed seeing)')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/seeing_nc/future_trends/wind_speed/"
  ylabel <- 'Astronomical seeing [arcesc]'
  cpressure <- c('_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level')
  unit <- 'arcsec'
  offset <- 0.6
  
  
} else if (variable == 'TCW'){
  print('precipitable water vapor')
  base_path = "./chaldene/Astroclimate_Project/Model_evaluation/TCW/future_trends/"
  cpressure <- c('_600', '_700', '_700', '_850','_700', '_925', '_850', '_700')
  ylabel <- 'PWV [mmH2O]'
  unit <- 'mmH2O'
  offset <- 6
} 


###### MAIN

# def.par <- par(no.readonly = TRUE) # save default, for resetting...
# 
# ## divide the device into two rows and two columns
# ## allocate figure 1 all of row 1
# ## allocate figure 2 the intersection of column 2 and row 2
# layout(matrix(c(1,2,3,3,3,4,4,4), 8, 1, byrow = TRUE), heights=c(1,1,2,2))
# ## show the regions that have been allocated to each plot
# layout.show(4)

# ATTENTION: INdexing starts at 1 in R!

# [0]: Mauna Kea --> 1
# [1]: Cerro Paranal --> 2
# [2]: La Silla --> 3
# [3]: Cerro Tololo --> 4
# [4]: La Palma --> 5
# [5]: Siding Spring --> 6
# [6]: Sutherland --> 7
# [7]: SPM --> 8

csites <- c("Mauna_Kea", "Cerro_Paranal", "La_Silla", "Cerro_Tololo", "La_Palma", "Siding_Spring", "Sutherland", "SPM")
# csites <- c("Mauna_Kea", "Cerro_Paranal")
cforcing <- c('hist', 'present', 'SSTfuture','future') # 'future', 'hist', 
# cforcing <- c('future')

path_list <- c()

##### density plots
# density_plots_insitu(base_path, csites, cpressure)

############################# map2stan

# write.csv(my_df,  paste(base_path, 'Temperature_future_trends.csv' , sep=''), row.names=T)

for (forc in cforcing){
  
  pdf(width=8, height=18, pointsize=10, file=paste(base_path, variable, '_', forc, "_Linear_fit_monthly_timeseries_map2stan.pdf" , sep=''))
  par(mfrow=c(8,1)) # , mai=c(0.1,0.1,0.1,0.1)) 

  
  for (idx in seq_along(csites)) {
    par(mfrow=c(8,1))
    # path = paste(base_path, csites[idx], forc, cpressure[idx], "_5month_index_rolling_dataset.csv"  , sep='') # Siding_SpringSSTfuture_925_5month_index_rolling_dataset.csv
    path = paste(base_path, csites[idx],'_', forc,   cpressure[idx], "_monthly_timeseries.csv"  , sep='')
    
    print(path)
    # path_list <- append(path_list, path) 
    
    # read in data
    d <- read.csv(path, header=TRUE, sep=",")
    
    # define time with as.numeric(rownames(d))
    d$month_index <- as.numeric(rownames(d))
    
    if (variable == 'total_cloud_cover'){
      # convert decimal to percentages of cloud cover
      d$temp <- d$temp * 100
    } else if (variable == 'SH'){
      d$temp <- d$temp * 1000 # convert to g/kg
    }
    
    # define m_prior and a_prior (now if we have the rowname, it makes sense to set m_prior to 0 and a_prior to the mean value)
    m_prior = 0
    a_prior_mean = mean(d$temp)-m_prior
    print(a_prior_mean)
    
    # map (maximum a posteriori estimate; uses quadratic approximation)
    # what this blackbox actually does: 
    # we tell the model that the prior distribution is composed of a normal distribution with mu and sigma, mu being the linear regression line
    # sigma is the standard deviation of the normal distribution of the prior
    # sigma is retrieved from a uniform distribution. e.g. if sigma ~ dunif(0 , 10), this means that 95% of the variables lie within 20 units around the mean value
    # mu is retrieved from a normal distribution
    # you can plot the prior distribution of mu with curve( dnorm( x , mu , sigma ) , from=x_start , to=x_end )
    # plot the prior distribution of the variable with
    # sample_mu <- rnorm( 1e4 , 178 , 20 )
    # sample_sigma <- runif( 1e4 , 0 , 50 )
    # prior_h <- rnorm( 1e4 , sample_mu , sample_sigma )
    # dens( prior_h )
    # extract samples from the model: 
    # post <- extract.samples( m1 )
    # post[1:5,]
    
    # trim data
    d.trim <- d[ , c("temp","month_index") ]
    d.trim$a_prior_mean <- a_prior_mean # add a_prior_mean so it can be used!
    
    m1 <- NULL
    attempt <- 0
    
    while( is.null(m1) && attempt <= 3 ) {
      attempt <- attempt + 1
      try(
        # if it fails, it is still null, otherwise it is not null
        
        m1 <- map2stan(
          alist(
            temp ~ dnorm(  mu , sigma ) ,
            mu <- a + b*month_index + Amp*sin(2*3.14159265359/12 * ( month_index + x)), # p. 113 # map2stan did not regognize pi, so I use 3.14159265359
            #a ~ dnorm( mean(d$temp) , 5 ) ,
            a ~ dnorm( a_prior_mean , 5) ,
            b ~ dnorm( 0 , 3) ,
            Amp ~ dnorm(0, 6),
            x ~ dnorm(0 , 6),
            sigma ~ dunif( 0 , 10 )
          ) ,
          data=d.trim ) #,
        # start=list(a=a_prior_mean))
        
      )
    } 
    
    print(precis( m1 , digits = 5, corr=TRUE))
    pr1 <- precis(m1) # access with pr1['b', 'mean']
    
    mu <- link( m1 , data=data.frame(month_index=d$month_index) )
    
    mu.mean <- apply( mu , 2 , mean )
    mu.HPDI <- apply( mu , 2 , HPDI , prob=0.89 )
    
    sim.temp <- sim( m1 , data=list(month_index=d$month_index), n=1e4 ) # increasing number of samples smooths out the intervals
    str(sim.temp)
    
    temp.PI <- apply( sim.temp , 2 , PI , prob=0.89 )
    
    # calculate residuals
    muca <- coef(m1)['a'] + coef(m1)['b']*d$month_index + coef(m1)['Amp']*sin(2*pi/12 * ( d$month_index + coef(m1)['x']))
    # compute residual for each State (in percent)
    
    m.resid <- (d$temp - muca) # how to properly normalize?
    
   
    # write.csv(summary(m1),  paste(base_path, csites[idx],'_', forc, cpressure[idx], '_TESTING.csv' , sep=''), row.names=T)
    
    # plot everything
    
    # plot text (and save to .csv for latex Table)
    # su1 <- summary(m1)
    # extract with su1@output$Mean[1] (1 for a, 2 for b, 3 for sigma)
    bstr1 <- pr1['b', 'mean']
    stdstr1 <- pr1['b', 'sd']
    bstr55 <- pr1['b', '5.5%']
    bstr945 <- pr1['b', '94.5%']
    
    # *12 to have it per year
    bstr1.2 <- formatC(bstr1 * 12, digits=3,format="fg", flag="#")
    stdstr1.2 <- formatC(stdstr1 * 12, digits=2,format="fg", flag="#")
    bstr55.2 <- formatC(bstr55 * 12, digits=3,format="fg", flag="#")
    bstr945.2 <- formatC(bstr945 * 12, digits=3,format="fg", flag="#")
    
    plot( temp ~ month_index , data=d , col=col.alpha(rangi2,0.8), ylim= c(mean(d$temp)-offset, mean(d$temp)+offset),  cex=1, pch=19 , ylab = ylabel,font.main = 1, main=paste(csites[idx], forc, cpressure[idx] , sep=' ')) #
    mtext(paste('b = ', bstr1.2," \u00B1 ",stdstr1.2, ' ', unit, '/year\n89%-Interval = (', bstr55.2, ', ',bstr945.2,') ' ,unit ,'/year', sep=''), 
          side=3, line=-2, cex=0.7)
    
    lines( d$month_index , mu.mean )
    # plot a shaded region for 89% HPDI (highest posterior densitiy interval)
    shade( mu.HPDI  ,d$month_index )
    # draw PI (percentile intervals) region for simulated temp values
    shade( temp.PI , d$month_index )
    
    # plot residuals
    plot(d$month_index, m.resid)
    
    # plot markov chain
    plot(m1)
    
    # plot pairs
    pairs(m1)
    
    # write output to .csv
    write.csv(pr1,  paste(base_path, csites[idx],'_', forc, cpressure[idx],  variable , '_', '_PRIMAVERA_Projections_Bayesian_model_map2stan.csv' , sep=''), row.names=T)
   
  }
  dev.off() # close file
}

