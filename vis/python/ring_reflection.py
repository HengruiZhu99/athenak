"""Identify the physical equatorial boundary in meridional ring snapshots."""


def reflected_equator(frame):
    mesh = False
    for raw in frame["header"]:
        line = raw.split("#", 1)[0].strip()
        if line.startswith("<"):
            mesh = line == "<mesh>"
        if mesh and line.partition("=")[0].strip() == "ix2_bc":
            return (line.partition("=")[2].strip() == "reflect"
                    and min(block[2] for block in frame["mb_geometry"]) == 0)
    return False
