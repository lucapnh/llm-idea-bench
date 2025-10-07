# Title: Gaussian Approximation Potentials: a brief tutorial

## Keywords

Gaussian process regression; interatomic potentials; SOAP descriptors; machine learning for materials; QUIP

## TL;DR

The paper introduces Gaussian Approximation Potentials (GAP), a Gaussian-process framework that learns accurate, local interatomic energy functions from quantum-mechanical energies and forces using symmetry-aware descriptors such as SOAP, with practical guidance and an example implementation in QUIP.&#x20;

## Abstract

This tutorial paper presents the Gaussian Approximation Potentials (GAP) framework for fitting interatomic potentials directly to quantum-mechanical reference data while respecting the locality and symmetries of atomic environments. GAP models the total energy as a sum of short-range, atom-centered contributions (optionally augmented by long-range terms) and learns these local energy functionals via Gaussian process regression over carefully designed descriptors. The authors detail how to train on total energies and their derivatives (forces and virials), derive the requisite covariance expressions, and incorporate realistic aspects of atomistic modeling: compact cutoffs, data noise, multi-body decompositions (e.g., separate 2- and 3-body GPs), and computational sparsification using representative environments. A central contribution is the Smooth Overlap of Atomic Positions (SOAP) descriptor and its associated rotationally and permutationally invariant kernel, which provide a robust similarity measure between atomic neighborhoods. The paper also outlines alternative descriptor choices (pairs, triplets, molecular dimers) and shows how permutation symmetry can be enforced at the kernel level. Practically, the methodology is implemented in the QUIP software, with examples of kernel choices, hyperparameterization, and command-line workflows. A didactic case study demonstrates that GAP trained only on total energies and forces can faithfully reconstruct the two- and three-body terms of the Stillinger–Weber silicon potential, validating the approach while highlighting the importance of coverage near cutoffs. Overall, GAP offers a principled, data-efficient route to high-fidelity, transferable interatomic models suitable for large-scale simulation.&#x20;
