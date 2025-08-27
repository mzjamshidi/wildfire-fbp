from fbp import FBP_FUEL_DESC

def describe_fbp_fuel_types():
    print("="*50)
    for fuel, desc in FBP_FUEL_DESC.items():
        print(f"{fuel:>5} : {desc:<50}")
    print("="*50)