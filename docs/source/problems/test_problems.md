---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_test_problems:
```

# Test Problems

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. toctree::
   :hidden:
   :maxdepth: 1
   
   single/ackley.ipynb
   single/griewank.ipynb
   single/zakharov.ipynb
   single/rastrigin.ipynb
   single/rosenbrock.ipynb

   multi/bnh.ipynb
   multi/zdt.ipynb
   multi/osy.ipynb
   multi/tnk.ipynb
   multi/truss2d.ipynb
   multi/welded_beam.ipynb
   multi/omni_test.ipynb
   multi/sym_part.ipynb

   many/dtlz.ipynb
   many/wfg.ipynb

   constrained/mw.ipynb
   constrained/dascmop.ipynb
   constrained/modact.ipynb
   
   dynamic/df.ipynb
```

In the future, we are planning to provide a comprehensive overview of the problems. So far, we have managed to describe a few of them and plot the functions. Please note that we have used for some problems the descriptions from [here](https://www.sfu.ca/~ssurjano/).

We want to keep the function definitions as error-free as possible and matching with the implementation. If you find any deviation, please let us know.

+++

Various test problems are already implemented and available by providing the corresponding problem name we have assigned to it. A couple of problems can be further parameterized by providing the number of variables, constraints, or other problem-dependent constants.

```{code-cell} ipython3
from pymoo.problems import get_problem

# create a simple test problem from string
p = get_problem("Ackley")

# the input name is not case sensitive
p = get_problem("ackley")

# also input parameter can be provided directly
p = get_problem("dtlz1^-1", n_var=20, n_obj=5)
```

## Many-Objective

+++

|Problem|Description|
|-|-|
|[DTLZ1](many/dtlz.ipynb#dtlz1)|many-objective|
|[DTLZ2](many/dtlz.ipynb#dtlz2)||
|[DTLZ3](many/dtlz.ipynb#dtlz3)||
|[DTLZ4](many/dtlz.ipynb#dtlz4)||
|[DTLZ5](many/dtlz.ipynb#dtlz5)||
|[DTLZ6](many/dtlz.ipynb#dtlz6)||
|[DTLZ7](many/dtlz.ipynb#dtlz7)||
|DTLZ1^-1||
|Scaled DTLZ||
|Convex DTLZ||
|WFG1||
|WFG2||
|WFG3||
|WFG4||
|WFG5||
|WFG6||
|WFG7||
|WFG8||
|WFG9||

+++

## Multi-Objective

+++

|Problem|Variables|Objectives|Constraints|Description
|:---|:---|:---|:---|:---|
|[BNH](multi/bnh.ipynb)|2|2|2||
|Carside|7|3|10||
|Kursawe|3|2|&nbsp;||
|[OSY](multi/osy.ipynb)|6|2|6||
|[TNK](multi/tnk.ipynb)|2|2|2||
|[Truss2D](multi/truss2d.ipynb)|3|2|1||
|[Welded Beam](multi/welded_beam.ipynb)|4|2|4||
|CTP1|2|2|s||
|CTP2|2|2|s||
|CTP3|2|2|s||
|CTP4|2|2|s||
|CTP5|2|2|s||
|CTP6|2|2|s||
|CTP7|2|2|s||
|CTP8|2|2|s||
|[ZDT1](multi/zdt.ipynb#zdt1)|30|2|&nbsp;||
|[ZDT2](multi/zdt.ipynb#zdt2)|30|2|&nbsp;||
|[ZDT3](multi/zdt.ipynb#zdt3)|30|2|&nbsp;||
|[ZDT4](multi/zdt.ipynb#zdt4)|10|2|&nbsp;||
|[ZDT5](multi/zdt.ipynb#zdt5)|80|2|&nbsp;||
|[ZDT6](multi/zdt.ipynb#zdt6)|10|2|&nbsp;||
|[DASCMOP1](constrained/dascmop.ipynb)|30|2|11||
|[DASCMOP2](constrained/dascmop.ipynb)|30|2|11||
|[DASCMOP3](constrained/dascmop.ipynb)|30|2|11||
|[DASCMOP4](constrained/dascmop.ipynb)|30|2|11||
|[DASCMOP5](constrained/dascmop.ipynb)|30|2|11||
|[DASCMOP6](constrained/dascmop.ipynb)|30|2|11||
|[DASCMOP7](constrained/dascmop.ipynb)|30|3|7||
|[DASCMOP8](constrained/dascmop.ipynb)|30|3|7||
|[DASCMOP9](constrained/dascmop.ipynb)|30|3|7||
|[MW1](constrained/mw.ipynb)|15|2|1||
|[MW2](constrained/mw.ipynb)|15|2|1||
|[MW3](constrained/mw.ipynb)|15|2|2||
|[MW4](constrained/mw.ipynb)|15|3|1||
|[MW5](constrained/mw.ipynb)|15|2|3||
|[MW6](constrained/mw.ipynb)|15|2|1||
|[MW7](constrained/mw.ipynb)|15|2|2||
|[MW8](constrained/mw.ipynb)|15|3|1||
|[MW9](constrained/mw.ipynb)|15|2|1||
|[MW10](constrained/mw.ipynb)|15|2|3||
|[MW11](constrained/mw.ipynb)|15|2|4||
|[MW12](constrained/mw.ipynb)|15|2|2||
|[MW13](constrained/mw.ipynb)|15|2|2||
|[MW14](constrained/mw.ipynb)|15|3|1||
|[SymPart](multi/sym_part.ipynb)|2|2|&nbsp;||
|[OmniTest](multi/omni_test.ipynb)|s|2|&nbsp;||
|[MODAct](constrained/modact.ipynb)|20|2-5|7-10|Real-world mechanical design problems|

+++

## Single-Objective

+++

|Problem|Variables|Constraints|Type|
|-|-|-|-|
|[Ackley](single/ackley.ipynb)|s|&nbsp;||
|Cantilevered Beams|4|2||
|[Griewank](single/griewank.ipynb)|s|&nbsp;||
|Himmelblau|2|&nbsp;||
|Knapsack|s|1||
|Pressure Vessel|4|4||
|[Rastrigin](single/rastrigin.ipynb)|s|&nbsp;||
|[Rosenbrock](single/rosenbrock.ipynb)|s|&nbsp;||
|Schwefel|s|&nbsp;||
|Sphere|s|&nbsp;||
|[Zakharov](single/zakharov.ipynb)|s|&nbsp;||
|G01|13|9||
|G02|20|2||
|G03|10|1||
|G04|5|6||
|G05|4|5||
|G06|2|2||
|G07|10|8||
|G08|2|2||
|G09|8|6||

+++

**Global Optimization** 

Implementations are taken from [SciPy](https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions) where a variety of global optimization benchmark problems are provided. Pymoo uses a wrapper class to make all those functions available. 

+++

|Problem|Variables|Constraints|Name|
|-|-|-|-|
|AMGM|2||"go-amgm"|
|Ackley01|2||"go-ackley01"|
|Ackley02|2||"go-ackley02"|
|Ackley03|2||"go-ackley03"|
|Adjiman|2||"go-adjiman"|
|Alpine01|2||"go-alpine01"|
|Alpine02|2||"go-alpine02"|
|BartelsConn|2||"go-bartelsconn"|
|Beale|2||"go-beale"|
|BiggsExp02|2||"go-biggsexp02"|
|BiggsExp03|3||"go-biggsexp03"|
|BiggsExp04|4||"go-biggsexp04"|
|BiggsExp05|5||"go-biggsexp05"|
|Bird|2||"go-bird"|
|Bohachevsky1|2||"go-bohachevsky1"|
|Bohachevsky2|2||"go-bohachevsky2"|
|Bohachevsky3|2||"go-bohachevsky3"|
|BoxBetts|3||"go-boxbetts"|
|Branin01|2||"go-branin01"|
|Branin02|2||"go-branin02"|
|Brent|2||"go-brent"|
|Brown|2||"go-brown"|
|Bukin02|2||"go-bukin02"|
|Bukin04|2||"go-bukin04"|
|Bukin06|2||"go-bukin06"|
|CarromTable|2||"go-carromtable"|
|Chichinadze|2||"go-chichinadze"|
|Cigar|2||"go-cigar"|
|Cola|17||"go-cola"|
|Colville|4||"go-colville"|
|Corana|4||"go-corana"|
|CosineMixture|2||"go-cosinemixture"|
|CrossInTray|2||"go-crossintray"|
|CrossLegTable|2||"go-crosslegtable"|
|CrownedCross|2||"go-crownedcross"|
|Csendes|2||"go-csendes"|
|Cube|2||"go-cube"|
|Damavandi|2||"go-damavandi"|
|DeVilliersGlasser01|4||"go-devilliersglasser01"|
|DeVilliersGlasser02|5||"go-devilliersglasser02"|
|Deb01|2||"go-deb01"|
|Deb03|2||"go-deb03"|
|Decanomial|2||"go-decanomial"|
|Deceptive|2||"go-deceptive"|
|DeckkersAarts|2||"go-deckkersaarts"|
|DeflectedCorrugatedSpring|2||"go-deflectedcorrugatedspring"|
|DixonPrice|2||"go-dixonprice"|
|Dolan|5||"go-dolan"|
|DropWave|2||"go-dropwave"|
|Easom|2||"go-easom"|
|Eckerle4|3||"go-eckerle4"|
|EggCrate|2||"go-eggcrate"|
|EggHolder|2||"go-eggholder"|
|ElAttarVidyasagarDutta|2||"go-elattarvidyasagardutta"|
|Exp2|2||"go-exp2"|
|Exponential|2||"go-exponential"|
|FreudensteinRoth|2||"go-freudensteinroth"|
|Gear|4||"go-gear"|
|Giunta|2||"go-giunta"|
|GoldsteinPrice|2||"go-goldsteinprice"|
|Griewank|2||"go-griewank"|
|Gulf|3||"go-gulf"|
|Hansen|2||"go-hansen"|
|Hartmann3|3||"go-hartmann3"|
|Hartmann6|6||"go-hartmann6"|
|HelicalValley|3||"go-helicalvalley"|
|HimmelBlau|2||"go-himmelblau"|
|HolderTable|2||"go-holdertable"|
|Hosaki|2||"go-hosaki"|
|Infinity|2||"go-infinity"|
|JennrichSampson|2||"go-jennrichsampson"|
|Judge|2||"go-judge"|
|Katsuura|2||"go-katsuura"|
|Keane|2||"go-keane"|
|Kowalik|4||"go-kowalik"|
|Langermann|2||"go-langermann"|
|LennardJones|6||"go-lennardjones"|
|Leon|2||"go-leon"|
|Levy03|2||"go-levy03"|
|Levy05|2||"go-levy05"|
|Levy13|2||"go-levy13"|
|Matyas|2||"go-matyas"|
|McCormick|2||"go-mccormick"|
|Meyer|3||"go-meyer"|
|Michalewicz|2||"go-michalewicz"|
|MieleCantrell|4||"go-mielecantrell"|
|Mishra01|2||"go-mishra01"|
|Mishra02|2||"go-mishra02"|
|Mishra03|2||"go-mishra03"|
|Mishra04|2||"go-mishra04"|
|Mishra05|2||"go-mishra05"|
|Mishra06|2||"go-mishra06"|
|Mishra07|2||"go-mishra07"|
|Mishra08|2||"go-mishra08"|
|Mishra09|3||"go-mishra09"|
|Mishra10|2||"go-mishra10"|
|Mishra11|2||"go-mishra11"|
|MultiModal|2||"go-multimodal"|
|NeedleEye|2||"go-needleeye"|
|NewFunction01|2||"go-newfunction01"|
|NewFunction02|2||"go-newfunction02"|
|OddSquare|2||"go-oddsquare"|
|Parsopoulos|2||"go-parsopoulos"|
|Pathological|2||"go-pathological"|
|Paviani|10||"go-paviani"|
|PenHolder|2||"go-penholder"|
|Penalty01|2||"go-penalty01"|
|Penalty02|2||"go-penalty02"|
|PermFunction01|2||"go-permfunction01"|
|PermFunction02|2||"go-permfunction02"|
|Pinter|2||"go-pinter"|
|Plateau|2||"go-plateau"|
|Powell|4||"go-powell"|
|PowerSum|4||"go-powersum"|
|Price01|2||"go-price01"|
|Price02|2||"go-price02"|
|Price03|2||"go-price03"|
|Price04|2||"go-price04"|
|Qing|2||"go-qing"|
|Quadratic|2||"go-quadratic"|
|Quintic|2||"go-quintic"|
|Rana|2||"go-rana"|
|Rastrigin|2||"go-rastrigin"|
|Ratkowsky01|4||"go-ratkowsky01"|
|Ratkowsky02|3||"go-ratkowsky02"|
|Ripple01|2||"go-ripple01"|
|Ripple25|2||"go-ripple25"|
|Rosenbrock|2||"go-rosenbrock"|
|RosenbrockModified|2||"go-rosenbrockmodified"|
|RotatedEllipse01|2||"go-rotatedellipse01"|
|RotatedEllipse02|2||"go-rotatedellipse02"|
|Salomon|2||"go-salomon"|
|Sargan|2||"go-sargan"|
|Schaffer01|2||"go-schaffer01"|
|Schaffer02|2||"go-schaffer02"|
|Schaffer03|2||"go-schaffer03"|
|Schaffer04|2||"go-schaffer04"|
|Schwefel01|2||"go-schwefel01"|
|Schwefel02|2||"go-schwefel02"|
|Schwefel04|2||"go-schwefel04"|
|Schwefel06|2||"go-schwefel06"|
|Schwefel20|2||"go-schwefel20"|
|Schwefel21|2||"go-schwefel21"|
|Schwefel22|2||"go-schwefel22"|
|Schwefel26|2||"go-schwefel26"|
|Schwefel36|2||"go-schwefel36"|
|Shekel05|4||"go-shekel05"|
|Shekel07|4||"go-shekel07"|
|Shekel10|4||"go-shekel10"|
|Shubert01|2||"go-shubert01"|
|Shubert03|2||"go-shubert03"|
|Shubert04|2||"go-shubert04"|
|SineEnvelope|2||"go-sineenvelope"|
|SixHumpCamel|2||"go-sixhumpcamel"|
|Sodp|2||"go-sodp"|
|Sphere|2||"go-sphere"|
|Step|2||"go-step"|
|Step2|2||"go-step2"|
|Stochastic|2||"go-stochastic"|
|StretchedV|2||"go-stretchedv"|
|StyblinskiTang|2||"go-styblinskitang"|
|TestTubeHolder|2||"go-testtubeholder"|
|ThreeHumpCamel|2||"go-threehumpcamel"|
|Thurber|7||"go-thurber"|
|Treccani|2||"go-treccani"|
|Trefethen|2||"go-trefethen"|
|Trid|6||"go-trid"|
|Trigonometric01|2||"go-trigonometric01"|
|Trigonometric02|2||"go-trigonometric02"|
|Tripod|2||"go-tripod"|
|Ursem01|2||"go-ursem01"|
|Ursem03|2||"go-ursem03"|
|Ursem04|2||"go-ursem04"|
|UrsemWaves|2||"go-ursemwaves"|
|VenterSobiezcczanskiSobieski|2||"go-ventersobiezcczanskisobieski"|
|Vincent|2||"go-vincent"|
|Watson|6||"go-watson"|
|Wavy|2||"go-wavy"|
|WayburnSeader01|2||"go-wayburnseader01"|
|WayburnSeader02|2||"go-wayburnseader02"|
|Weierstrass|2||"go-weierstrass"|
|Whitley|2||"go-whitley"|
|Wolfe|3||"go-wolfe"|
|XinSheYang01|2||"go-xinsheyang01"|
|XinSheYang02|2||"go-xinsheyang02"|
|XinSheYang03|2||"go-xinsheyang03"|
|XinSheYang04|2||"go-xinsheyang04"|
|Xor|9||"go-xor"|
|YaoLiu04|2||"go-yaoliu04"|
|YaoLiu09|2||"go-yaoliu09"|
|Zacharov|2||"go-zacharov"|
|ZeroSum|2||"go-zerosum"|
|Zettl|2||"go-zettl"|
|Zimmerman|2||"go-zimmerman"|
|Zirilli|2||"go-zirilli"|
