
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
                                <a href="problems/index.html">Single-objective</a>,
                                <a href="problems/index.html">Multi-objective</a>,
                                <a href="problems/index.html">Many-objective</a></br>
                                <a href="problems/index.html">Custom Problem</a>,
                                <a href="problems/index.html">Gradients</a>
                                <a href="problems/index.html">Parallelization</a>
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
                                <a href="operators/sampling.html#nb-sampling-random">Random</a>,
                                <a href="operators/sampling.html#nb-sampling-lhs">LHS</a>
                                </br>
                            <a href="operators/selection.html">Selection:</a>
                                <a href="operators/sampling.html#nb-selection-random">Random</a>,
                                <a href="operators/sampling.html#nb-selection-tournament">Binary Tournament</a>
                                </br>
                            <a href="operators/crossover.html">Crossover:</a>
                                <a href="operators/crossover.html#nb-crossover-random">SBX</a>,
                                <a href="operators/crossover.html#nb-crossover-tournament">UX</a>,
                                <a href="operators/crossover.html#nb-crossover-random">HUX</a>,
                                <a href="operators/crossover.html#nb-crossover-tournament">DE</a>,
                                <a href="operators/crossover.html#nb-crossover-tournament">Point</a>,
                                <a href="operators/crossover.html#nb-crossover-tournament">Exponential</a>
                                </br>
                            
                            <a href="operators/mutation.html">Mutation:</a>
                                <a href="operators/crossover.html#nb-mutation-tournament">Polynomial</a>,
                                <a href="operators/crossover.html#nb-crossover-tournament">Bitflip</a>
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
                            <h4>Visualization</h4>
                            <a href="/visualization/scatter.html">Scatter Plot (2D/3D/ND)</a>,
                            <a href="/problems/pcp.html">Parallel Coordinate Plot (PCP) </a>,
                            <a href="/problems/radviz.html">Radviz</a>,
                            <a href="/problems/star.html">Star Coordinates</a>,
                            <a href="/problems/heatmap.html">Heatmap</a>,
                            <a href="/problems/petal_width.html">Petal Width</a>,
                            <a href="/problems/radar.html">Spider Web / Radar</a>

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
                            <h4>Decision Making</h4>
                            <a href="operators/sampling.html">Compromise Programming</a>,
                            <a href="operators/sampling.html">Pseudo Weights</a>,
                            <a href="operators/sampling.html">Knee Point</a>
                    </div>

                </div>
            </div>

            <div class="entry col" onclick="location.href='visualization/index.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-medal fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Performance Indicator</h4>
                            <a href="performance_indicator/scatter.html">GD</a>,
                            <a href="performance_indicator/scatter.html">GD+</a>,
                            <a href="performance_indicator/scatter.html">IGD</a>,
                            <a href="performance_indicator/scatter.html">IGD+</a>,
                            <a href="performance_indicator/scatter.html">Hypervolume</a>
                    </div>
                </div>
            </div>

        </div>




        <div class="row row-eq-height">

            <div class="entry col" onclick="location.href='decision_making/index.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-layer-group fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Decomposition</h4>
                            <a href="operators/sampling.html">Weighted-Sum</a>,
                            <a href="operators/sampling.html">ASF</a>,
                            <a href="operators/sampling.html">AASF</a>,
                            <a href="operators/sampling.html">Tchebicheff</a>,
                            <a href="operators/sampling.html">PBI</a>
                    </div>

                </div>
            </div>

            <div class="entry col" onclick="location.href='visualization/index.html';">
                <div class="d-flex flex-row">
                    <div class="icon col-2">
                        <i class="fas fa-blender fa-2x"></i>
                    </div>
                    <div class="desc col-10">
                            <h4>Miscellaneous</h4>
                            <a href="performance_indicator/scatter.html">Termination Criteria</a>,
                            <a href="performance_indicator/scatter.html">Reference Directions</a>
                    </div>
                </div>
            </div>

        </div>


    </div>