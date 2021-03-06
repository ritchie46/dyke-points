import glob
import math

CPOINT_CHECK_COLCOUNT = 55
CPOINT_NAMES = {
    0: 'maaiveld binnenwaarts',
    1: 'insteek sloot polderzijde',
    2: 'slootbodem polderzijde',
    3: 'slootbodem dijkzijde',
    4: 'insteek sloot dijkzijde',
    5: 'teen dijk binnenwaarts',
    6: 'kruin binnenberm',
    7: 'insteek binnenberm',
    8: 'kruin binnentalud',
    9: 'verkeersbelasting kant binnenwaarts',
    10: 'verkeersbelasting kant buitenwaarts',
    11: 'kruin buitentalud',
    12: 'insteek buitenberm',
    13: 'kruin buitenberm',
    14: 'teen dijk buitenwaarts',
    15: 'insteek geul',
    16: 'teen geul',
    17: 'maaiveld buitenwaarts'
}


def get_all_surfacelines(waterschap):
    """Read the data folder and collect all surfacelines.csv files

    Unfortunately every waterboard seems to have its own idea of the structure of the csv files...
    * rijnland: uses 2D points (x,z)

    Input : waterschap name (rijnland, ...)
    Output: dict containing {}id: points} where id = str and points = list of (CPOINT_ID,x,z) tuples
            since the surfaceline does not yet have characteristic points defined all CPOINT_IDs are
            set to -1
            example ("my_crs_id":[(-1,x0,z0),(-1,x1,z1),...,(-1,xn,zn)])
    """
    result = {}
    if waterschap == "rijnland":
        slinefiles = glob.glob("../data/rijnland/**/surfacelines.csv", recursive=True)
        for f in slinefiles:
            lines = open(f, 'r').readlines()[1:]

            col_x0 = 5
            col_z0 = 7
            if f.find("Nieuwkoop") > -1:
                col_x0 = 1
                col_z0 = 3

            for line in lines:
                args = line.split(';')
                id = args[0]
                xs = [float(x) for x in args[col_x0::3] if len(x.strip()) > 0]
                zs = [float(z) for z in args[col_z0::3] if len(z.strip()) > 0]

                if xs[0] == 0.0:
                    cid = [-1] * len(xs)
                    points = list(zip(cid, xs, zs))
                    result[id] = points
                else:
                    print("Dwarsprofiel %s begint niet op x=0 en wordt geskipped" % id)
    else:
        print("Dit waterschap is nog niet bekend.")
        return result

    return result


def get_all_cpoints(waterschap):
    """Read the data folder and collect all characteristicpoints.csv slinefiles

    Unfortunately every waterboard seems to have its own idea of the structure of the csv files (again ;-)...
    * rijnland: uses 2D points (x,z)

    We expect a 55 column (18 different characteristic points) file (might change in the future)

    Input : None
    Output: dict containing (id:cpoints) where id = str and cpoints = list of (CPOINT_ID, x, z)
            example ("my_crs_id":[(0,x0,y0),(1,x1,z1)...(17,x17,z17)])
            note that you can use the CPOINT_NAMES dictionary to find a meaningful name for the id
    """

    result = {}
    if waterschap == "rijnland":
        cpointfiles = glob.glob("../data/rijnland/**/characteristicpoints.csv", recursive=True)
        for f in cpointfiles:
            lines = open(f, 'r').readlines()
            colcount = len([a for a in lines[0].split(';') if len(a.strip()) > 0])

            if colcount != CPOINT_CHECK_COLCOUNT:
                print("Bestand %s bevat niet het aantal verwachten kolommen (%d).. skippen dus!" % (
                f, CPOINT_CHECK_COLCOUNT))
            else:
                for line in lines[1:]:
                    args = line.split(';')
                    id = args[0]
                    xs = [float(x) for x in args[1::3] if len(x.strip()) > 0]
                    zs = [float(z) for z in args[3::3] if len(z.strip()) > 0]
                    points = list(zip(CPOINT_NAMES.keys(), xs, zs))
                    result[id] = points
    else:
        print("Dit waterschap is nog niet bekend.")
        return result

    return result


def get_zmin(crosssection):
    zs = [p[2] for p in crosssection]
    return min(zs)


def get_zmax(crosssection):
    zs = [p[2] for p in crosssection]
    return max(zs)


def get_z_at(crosssection, xloc):
    xs = [p[1] for p in crosssection]
    zs = [p[2] for p in crosssection]
    if xloc < 0 or xloc > xs[-1]: return 1e-9

    for i in range(1, len(xs)):
        x1 = xs[i - 1]
        x2 = xs[i]
        if x1 <= xloc <= x2:
            return zs[i - 1] + (xloc - x1) / (x2 - x1) * (zs[i] - zs[i - 1])

    print("Hier zou ik niet mogen komen")
    print("crosssection:", crosssection)
    print("xloc:", xloc)


def get_relative_height(z, crosssection):
    """Return the relative height (z) of the point where zmin=0.0 and zmax=1.0"""
    zmin = get_zmin(crosssection)
    zmax = get_zmax(crosssection)
    return (z - zmin) / (zmax - zmin)


def get_slope(p, dp, xoffset):
    """Return the angle from point p (px, pz) to point (px - xoffset, z(px-xoffset))"""
    x1, z1 = p[1], p[2]
    x2 = x1 + xoffset
    z2 = get_z_at(dp, x2)
    if z2 == -1e9:
        return 1e9
    else:
        alpha = math.atan2(z2 - z1, x2 - x1) * (180. / math.pi)
        if (alpha < 0): alpha = 360. + alpha
        return alpha


if __name__ == "__main__":
    sls = get_all_surfacelines("rijnland")
    cps = get_all_cpoints("rijnland")

    dp = sls['024_042_00002']
    print(get_z_at(dp, 110))
    print(get_slope(dp[3], dp, -2))
