/*! ======================================================================
 *
 *  \file SelectionMOO.h
 *
 *  \brief Implementation of several algorithms for calculating 
 *	the hypervolume of a set of points.
 * 
 *  \author Thomas Vo&szlig; <thomas.voss@rub.de>
 *
 *  \par 
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 * 
 *  \par Project:
 *      MOO-EALib
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Camâbridge, MA 02139, USA.
 */

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
