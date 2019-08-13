
.. raw:: html

    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.1/css/all.css" integrity="sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf" crossorigin="anonymous">

    <style>
    #wrapper {
        margin: 20px 25px; 

    }

    .entry {
        margin: 5px 2.5px 5px 2.5px;
        padding: 10px;
        border:1px solid #DCDCDC;
        height: auto;
    }

    .icon {
        margin-top: 5px;
    }

    .entry:hover {
        background: #DCDCDC;
        cursor: pointer;
    }


    </style>

    <div id="wrapper">
        <div class="row row-eq-height">
            <div class="entry col" onclick="location.href='algorithms/index.html';">
  
                    <div class="d-flex flex-row">
                        <div class="icon col-2">
                            <i class="fas fa-search fa-2x"></i>
                        </div>
                        <div class="desc col-10">
                                <h4>Algorithms</h4>
                                <a href="algorithms/genetic_algorithm.html">Genetic Algorithm</a>,
                                <a href="algorithms/differential_evolution.html">Differential Evolution</a>,
                                <a href="algorithms/nsga2.html">NSGA-II</a>,
                                <a href="algorithms/rnsga2.html">R-NSGA-II</a>,
                                <a href="algorithms/nsga3.html">NSGA-III</a>,
                                <a href="algorithms/rnsga3.html">R-NSGA-III</a>,
                                <a href="algorithms/unsga3.html">U-NSGA-III</a>,
                                <a href="algorithms/moead.html">MOEA/D</a>
                        </div>

                    </div>
            </div>

            <div class="entry col" onclick="location.href='problems/index.html';">
                    <div class="d-flex flex-row">
                        <div class="icon col-2">
                            <i class="fas fa-chess fa-2x"></i>
                        </div>
                        <div class="desc col-10">
                                <h4>Test Problems</h4>
                                <a href="problems/index.html#Single-Objective">Single-objective</a>,
                                <a href="problems/index.html#Multi-Objective">Multi-objective</a>,
                                <a href="problems/index.html#Many-Objective">Many-objective</a></br>
                                <a href="problems/custom.html">Problem Definition</a>,
                                <a href="problems/gradients.html">Gradients</a>
                        </div>
                    </div>
            </div>
        </div>




        <div class="row row-eq-height">
            <div class="entry col" onclick="location.href='operators/index.html';">

                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-tools fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Operators</h4>
                            <a href="operators/sampling.html">Sampling:</a>
                                Random, LHS
                                </br>
                            <a href="operators/selection.html">Selection:</a>
                                Random, Binary Tournament

                                </br>
                            <a href="operators/crossover.html">Crossover:</a>
                                SBX, UX, HUX, DE Point, Exponential
                                </br>
                            
                            <a href="operators/mutation.html">Mutation:</a>
                                Polynomial, Bitlfip
                                </br>

                    </div>

                 
                </div>
            </div>

            <div class="entry col" onclick="location.href='visualization/index.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-chart-line fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Visualization<img class="new-flag" src="_static/img/new_flag.svg"></h4>
                            <a href="/visualization/scatter.html">Scatter Plot (2D/3D/ND)</a>,
                            <a href="/visualization/pcp.html">Parallel Coordinate Plot (PCP) </a>,
                            <a href="/visualization/radviz.html">Radviz</a>,
                            <a href="/visualization/star.html">Star Coordinates</a>,
                            <a href="/visualization/heatmap.html">Heatmap</a>,
                            <a href="/visualization/petal.html">Petal Diagram</a>,
                            <a href="/visualization/radar.html">Spider Web / Radar</a>

                    </div>
                </div>
            </div>

            
        

        </div>

        <div class="row row-eq-height">

            <div class="entry col" onclick="location.href='decision_making/index.html';">

                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-balance-scale fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Decision Making<img class="new-flag" src="_static/img/new_flag.svg"></h4>
                            <a href="decision_making/index.html#nb-compromise">Compromise Programming</a>,
                            <a href="decision_making/index.html#nb-pseudo-weights">Pseudo Weights</a>,
                            <a href="decision_making/index.html#nb-high-tradeoff">High Trade-off Points</a>
                    </div>

                </div>
            </div>

            <div class="entry col" onclick="location.href='misc/performance_indicator.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-medal fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Performance Indicator</h4>
                            <a href="misc/performance_indicator.html#nb-gd">GD</a>,
                            <a href="misc/performance_indicator.html#nb-gd-plus">GD+</a>,
                            <a href="misc/performance_indicator.html#nb-igd">IGD</a>,
                            <a href="misc/performance_indicator.html#nb-igd-plus">IGD+</a>,
                            <a href="misc/performance_indicator.html#nb-hv">Hypervolume</a>
                    </div>
                </div>
            </div>

        </div>




        <div class="row row-eq-height">

            <div class="entry col" onclick="location.href='misc/decomposition.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-layer-group fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Decomposition</h4>
                            <a href="misc/decomposition.html#nb-weighted-sum">Weighted-Sum</a>,
                            <a href="misc/decomposition.html#nb-asf">ASF</a>,
                            <a href="misc/decomposition.html#nb-aasf">AASF</a>,
                            <a href="misc/decomposition.html#nb-tchebi">Tchebicheff</a>,
                            <a href="misc/decomposition.html#nb-pbi">PBI</a>
                    </div>

                </div>
            </div>

            <div class="entry col" onclick="location.href='misc/index.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-blender fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Miscellaneous</h4>
                            <a href="misc/termination_criterion.html">Termination Criterion</a>, 
                            <a href="misc/reference_directions.html">Reference Directions</a>, 
                            <a href="misc/constraint_handling.html">Constraint Handling</a>
                    </div>
                </div>
            </div>

        </div>



        <div class="row row-eq-height">

            <div class="entry col" onclick="location.href='tutorial/index.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-1">
                        <i class="fas fa-book-open fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Tutorials</h4>
                            Other Variable Types:
                            <a href="tutorial/binary_problem.html">Binary</a>,
                            <a href="tutorial/discrete_problem.html">Discrete</a>,
                            <a href="tutorial/mixed_variable_problem.html">Mixed</a>
                            </br>
                            More: <a href="problems/parallelization.html">Parallelization of Function Evaluations</a>
                    </div>

                </div>
            </div>
        </div>


    </div>