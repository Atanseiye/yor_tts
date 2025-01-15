

# List all installed packages
installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]

# Function to update a package
def update_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        print(f"Successfully updated {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update {package_name}: {e}")

# Update all packages
for package_name, _ in installed_packages:
    update_package(package_name)
