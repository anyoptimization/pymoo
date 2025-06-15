

#ifndef HYPERVOLUME_H
#define HYPERVOLUME_H

/*!
 *
 * \brief Implementation of the Overmars-Yap Algorithm, originally provided by N. Beume.
 *
 * Applicable to points of dimension d >= 3. It is assumed that the array of points is sorted
 * according to the 'last' objective.
 *
 * <PRE>
 * author = {Nicola Beume and G\"unther Rudolph},
 * title = {Faster {S}-Metric Calculation By Considering Dominated Hypervolume as {Klee}'s Measure Problem},
 * booktitle = {IASTED International Conference on Computational Intelligence},
 * publisher = {ACTA Press},
 * pages = {231-236},
 * year = {2006},
 * </PRE>
 */
 
double overmars_yap( double * points, double * referencePoint, unsigned noObjectives, unsigned noPoints );

/*!
 *
 * \brief Algorithm for the special case of d = 3 objectives.
 *
 * Applicable to points of dimension d = 3.
 *
 * Relevant literature:
 *
 * [1]  C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An
 *     improved dimension-sweep algorithm for the hypervolume
 *     indicator. In IEEE Congress on Evolutionary Computation,
 *     pages 1157-1163, Vancouver, Canada, July 2006.
 *
 * [2]  L. Paquete, C. M. Fonseca and M. Lopez-Ibanez. An optimal
 *     algorithm for a special case of Klee's measure problem in three
 *     dimensions. Technical Report CSI-RT-I-01/2006, CSI, Universidade
 *     do Algarve, 2006.
 */
 
double fonseca( double * points, double * referencePoint, unsigned noObjectives, unsigned noPoints );


//!
//! \brief computation of the hypervolume
//!
//! \par
//! This function acts as a frontend to various algorithms
//! for the computation of the hypervolume.
//!
//! \param  points          the list containes the coordinates of all points in a single array
//! \param  referencePoint  the reference or nadir point
//! \param  noObjectives    number of coordinates per point
//! \param  noPoints        number of points in the list
//! \return                 dominated hypervolume
//!
double hypervolume(double* points, double* referencePoint, unsigned int noObjectives, unsigned int noPoints);


#endif // HYPERVOLUME_H
