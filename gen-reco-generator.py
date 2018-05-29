import numpy

NUMEVENTS = 100      # exact number of events
AVENUMJETS = 10      # average number of jets per event
PHILOW = -numpy.pi   # bounding box of phi (azimuthal angle) and eta (~polar angle)
PHIHIGH = numpy.pi
ETALOW = -5
ETAHIGH = 5
ERRPHI = 0.01        # detector resolution
ERRETA = 0.01
RECOPROB = 0.95      # probability of not reconstructing a real jet
AVENUMFAKES = 1      # average number of spurious (fake) recontstructions

# simulate the generator-level jets
numgenjets = numpy.random.poisson(AVENUMJETS, NUMEVENTS)
genstops = numpy.cumsum(numgenjets)
genstarts = numpy.empty_like(genstops)
genstarts[0] = 0
genstarts[1:] = genstops[:-1]
genphi = numpy.random.uniform(PHILOW, PHIHIGH, genstops[-1])
geneta = numpy.random.uniform(ETALOW, ETAHIGH, genstops[-1])

# simulate mismeasurement (error in reconstructing phi and eta)
phiwitherr = genphi + numpy.random.normal(0, ERRPHI, genstops[-1])
etawitherr = geneta + numpy.random.normal(0, ERRETA, genstops[-1])

# simulate inefficiency in reconstruction (missing real jets)
recomask = (numpy.random.uniform(0, 1, genstops[-1]) < RECOPROB)

# simulate spurious (fake) jets per event
numfakes = numpy.random.poisson(AVENUMFAKES, NUMEVENTS)
fakestops = numpy.cumsum(numfakes)
fakestarts = numpy.empty_like(fakestops)
fakestarts[0] = 0
fakestarts[1:] = fakestops[:-1]
fakephi = numpy.random.uniform(PHILOW, PHIHIGH, fakestops[-1])
fakeeta = numpy.random.uniform(ETALOW, ETAHIGH, fakestops[-1])

# fill reconstructed data arrays
recostarts = numpy.empty_like(genstarts)
recostops = numpy.empty_like(genstops)
recophi = numpy.empty(recomask.sum() + numfakes.sum(), dtype=genphi.dtype)
recoeta = numpy.empty_like(recophi)

# (can't generate them in a vectorized way!)
truematches = []
recostart, recostop = 0, 0
for i in range(NUMEVENTS):
    genstart, genstop = genstarts[i], genstops[i]
    fakestart, fakestop = fakestarts[i], fakestops[i]
    mask = recomask[genstart:genstop]

    phi = phiwitherr[genstart:genstop][mask]    # generated phi with error and mask
    eta = etawitherr[genstart:genstop][mask]    # generated eta with error and mask

    # concatenate the subset of real jets with some fake jets
    holdphi = numpy.concatenate((phi, fakephi[fakestart:fakestop]))
    holdeta = numpy.concatenate((eta, fakeeta[fakestart:fakestop]))
    recostop += len(holdphi)

    # gen-level and reco-level data are both unordered sets; randomly permute
    order = numpy.random.permutation(recostop - recostart)
    recophi[recostart:recostop][order] = holdphi
    recoeta[recostart:recostop][order] = holdeta

    # keep that permutation to use as a "true match" map (not known to physicist!)
    truematch = numpy.ones(genstop - genstart, dtype=numgenjets.dtype) * -1
    truematch[mask] = order[:mask.sum()]
    truematches.append(truematch)

    recostarts[i] = recostart
    recostops[i] = recostop
    recostart = recostop

# print out the gen-level and reco-level jets: physicist doesn't know the "truematch"
for i in range(NUMEVENTS):
    unmatched = ["                             unmatched     ",
                 "{:>10s} {:>10s} {:>10s} {:>10s}".format("genphi", "geneta", "recophi", "recoeta"),
                 "-" * 44]
    matched = ["          matched",
               " {:>10s} {:>10s} {:>10s}".format("recophi", "recoeta", "deltaR"),
               "-" * 33]
    
    gphi = genphi[genstarts[i]:genstops[i]]
    geta = geneta[genstarts[i]:genstops[i]]
    rphi = recophi[recostarts[i]:recostops[i]]
    reta = recoeta[recostarts[i]:recostops[i]]

    # unmatched gen-level and reco-level side-by-side; either list may be longer than the other
    for j in range(max(len(gphi), len(rphi))):
        if j < len(gphi) and j < len(rphi):
            unmatched.append("{:10g} {:10g} {:10g} {:10g}".format(gphi[j], geta[j], rphi[j], reta[j]))
        elif j < len(gphi):
            unmatched.append("{:10g} {:10g} {:10s} {:10s}".format(gphi[j], geta[j], "", ""))
        else:
            unmatched.append("{:10s} {:10s} {:10g} {:10g}".format("", "", rphi[j], reta[j]))

    # for the gen-level jets that were truly measured, align with the true reco-level and show that deltaR is small
    for genj, recoj in enumerate(truematches[i]):
        if recoj == -1:
            matched.append(" {:10s} {:10s}".format("", ""))
        else:
            dphi = rphi[recoj] - gphi[genj]
            deta = reta[recoj] - geta[genj]
            while dphi < -numpy.pi:
                dphi += 2*numpy.pi
            while dphi >= numpy.pi:
                dphi -= 2*numpy.pi
            dr = numpy.sqrt(dphi**2 + deta**2)
            matched.append(" {:10g} {:10g} {:10g}".format(rphi[recoj], reta[recoj], dr))

    while len(matched) < len(unmatched):
        matched.append("")

    for one, two in zip(unmatched, matched):
        print("{}{}".format(one, two))
    print("")

# Example output:
#                              unmatched               matched
#     genphi     geneta    recophi    recoeta    recophi    recoeta     deltaR
# -----------------------------------------------------------------------------
# -0.0922835    2.22667   -2.44008    0.78393  -0.102562    2.23167  0.0114309
#   -2.42167   0.780564    1.95499  0.0570109   -2.44008    0.78393  0.0187209
# -0.0875004  0.0800505    2.65177   -2.48186 -0.0794158  0.0738131  0.0102111
#    0.26566     1.4085   -1.41866    4.35342   0.289994    1.41261  0.0246803
#    2.63594   -2.50299  -0.614436    2.94907    2.65177   -2.48186  0.0264005
#   -2.36245   -4.22215    3.03471   -1.32967                      
#   -1.42593    4.35982    2.87468    1.88595   -1.41866    4.35342 0.00968663
#   -1.73169   0.886889 -0.0794158  0.0738131   -1.72717   0.881607 0.00695575
#    3.04011   -1.31595  -0.102562    2.23167    3.03471   -1.32967  0.0147386
#    2.87492      1.868   -3.06057  0.0757893    2.87468    1.88595  0.0179481
#                         -1.72717   0.881607
#                         0.289994    1.41261
#
# a physicist would have to *discover* the truematch permutation array for every event by minimizing deltaR
