import gmsh
import sys

step = sys.argv[1]

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.occ.importShapes(step)
gmsh.model.occ.synchronize()

print("\n=== BEFORE HEALING ===")

for dim in range(4):
    entities = gmsh.model.getEntities(dim)
    print(f"dim={dim}: {len(entities)} entities")

print("\n3D entities:")
print(gmsh.model.getEntities(3))

print("\nHealing...")
gmsh.model.occ.healShapes(
    sewFaces=True,
    fixDegenerated=True,
    fixSmallEdges=True,
    fixSmallFaces=True,
    makeSolids=True,
    splitAngle=0.5
)

gmsh.model.occ.synchronize()

print("\n=== AFTER HEALING ===")

for dim in range(4):
    entities = gmsh.model.getEntities(dim)
    print(f"dim={dim}: {len(entities)} entities")

print("\n3D entities:")
print(gmsh.model.getEntities(3))

gmsh.finalize()