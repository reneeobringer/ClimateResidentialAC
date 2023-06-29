# ClimateResidentialAC

Code and data for an analysis of air conditioning use for 75 households across three US cities (Austin, TX, Ithaca, NY, and San Diego, CA). The results from the analysis have been accepted for publication in [_Socio-Economic Planning Sciences_](https://doi.org/10.1016/j.seps.2023.101664). The article can be cited as: 

```bibtex
@article{pezalla2023,
  title = {Evaluating the household-level climate-electricity nexus across three cities through statistical learning techniques},
  author = {Pezalla, Simon and Obringer, Renee},
  year = {2023},
  journal = {Socio-Economic Planning Sciences},
  volume = {},
  number = {},
  doi = {10.1016/j.seps.2023.101664}
}
```

The code was developed in R version 4.1.2 and last ran on 25 May 2023. `householdenergy.R` will run the model analysis. `varimp.R` is a supplementary R script to conduct variable importance analysis. `NRMSE_tidymodels.R` is a supplementary R script to calculate normalized root mean square error (NRMSE) within the tidymodels framework.

Two categories of data were collected: air conditioning data and climate data. The air conditioning data were collected from Pecan Street, Inc. Dataport (https://www.pecanstreet.org/dataport/) and are not included in this repository. Interested parties are encouraged to reach out to Peacn Street, Inc. for access. The climate data were obtained from the National Centers for Environmental Information, which maintains a database of weather station data. All data were collected in 2022.
