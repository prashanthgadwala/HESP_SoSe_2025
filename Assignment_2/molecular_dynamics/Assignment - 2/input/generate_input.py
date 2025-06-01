import os
import argparse
import math

def generate_particles(count, spacing=1.122):
    particles = []
    n = int(count ** (1/3)) + 1
    i = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if i >= count:
                    return particles
                px = 0.5 + x * spacing
                py = 0.5 + y * spacing
                pz = 0.5 + z * spacing
                particles.append(f"{px:.3f} {py:.3f} {pz:.3f}   0.0 0.0 0.0   1.0")
                i += 1
    return particles

def generate_collision_clusters(count_per_cluster=100, spacing=0.05, separation=1.5):
    cluster1 = generate_particles(count_per_cluster, spacing)
    cluster2 = generate_particles(count_per_cluster, spacing)

    # Offset cluster2 by separation in x direction
    cluster2 = [
        f"{float(p.split()[0]) + separation:.3f} {p.split()[1]} {p.split()[2]} 0.0 0.0 0.0 1.0"
        for p in cluster2
    ]
    return cluster1 + cluster2

def generate_repulsive_shell(points_on_shell=200, radius=1.0):
    # Distribute points roughly evenly on a sphere shell using Golden Spiral
    particles = []
    for i in range(points_on_shell):
        theta = 2 * math.pi * i / ((1 + math.sqrt(5)) / 2)  # golden angle approx
        z = 1 - (2 * i) / points_on_shell
        r = math.sqrt(1 - z*z)
        x = radius * r * math.cos(theta)
        y = radius * r * math.sin(theta)
        particles.append(f"{x:.3f} {y:.3f} {z*radius:.3f} 0.0 0.0 0.0 1.0")
    return particles

def generate_attractive_core(cluster_count=300, spacing=0.02):
    return generate_particles(cluster_count, spacing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, help="Generate cubic lattice particles with these counts")
    parser.add_argument("--case", type=str, choices=["collision_clusters", "repulsive_shell", "attractive_core"],
                        help="Generate special test case")
    parser.add_argument("--output", default="particles")
    # Parameters for cases
    parser.add_argument("--count_per_cluster", type=int, default=100, help="Particles per cluster for collision_clusters")
    parser.add_argument("--spacing", type=float, default=0.1, help="Particle spacing")
    parser.add_argument("--separation", type=float, default=1.5, help="Separation between clusters (collision_clusters)")
    parser.add_argument("--points_on_shell", type=int, default=200, help="Points on shell for repulsive_shell")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius of shell (repulsive_shell)")
    parser.add_argument("--cluster_count", type=int, default=300, help="Particles in core cluster (attractive_core)")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.case == "collision_clusters":
        particles = generate_collision_clusters(
            count_per_cluster=args.count_per_cluster,
            spacing=args.spacing,
            separation=args.separation
        )
        filepath = os.path.join(args.output, f"particles_collision_clusters_{len(particles)}.txt")
        with open(filepath, "w") as f:
            f.write("# x y z    vx vy vz    mass\n")
            f.write("\n".join(particles))
        print(f"Generated collision_clusters with {len(particles)} particles at {filepath}")

    elif args.case == "repulsive_shell":
        particles = generate_repulsive_shell(
            points_on_shell=args.points_on_shell,
            radius=args.radius
        )
        filepath = os.path.join(args.output, f"particles_repulsive_shell_{len(particles)}.txt")
        with open(filepath, "w") as f:
            f.write("# x y z    vx vy vz    mass\n")
            f.write("\n".join(particles))
        print(f"Generated repulsive_shell with {len(particles)} particles at {filepath}")

    elif args.case == "attractive_core":
        particles = generate_attractive_core(
            cluster_count=args.cluster_count,
            spacing=args.spacing
        )
        filepath = os.path.join(args.output, f"particles_attractive_core_{len(particles)}.txt")
        with open(filepath, "w") as f:
            f.write("# x y z    vx vy vz    mass\n")
            f.write("\n".join(particles))
        print(f"Generated attractive_core with {len(particles)} particles at {filepath}")

    else:
        # fallback: generate particles for all counts given
        if args.counts is None:
            print("Error: Provide --counts or --case")
            exit(1)

        for count in args.counts:
            particles = generate_particles(count, spacing=args.spacing)
            filepath = os.path.join(args.output, f"particles_{count}.txt")
            with open(filepath, "w") as f:
                f.write("# x y z    vx vy vz    mass\n")
                f.write("\n".join(particles))
            print(f"Generated {filepath} with {len(particles)} particles.")
