import os

# External path to the CO2 Dozen .states files
STATES_DIR = os.path.abspath(r"C:\Code\Work\raw_data_store\Triatomics\CO2")

# Define isotopes to loop through
ISOTOPES = [
    # Symmetric isotopologues
    {
        "id": "626",
        "C_mass": 12,
        "O_A_mass": 16,
        "O_B_mass": 16,
        "is_symmetric": 1,
        "file": "12C-16O2__Dozen.states.cut",
    },
    {
        "id": "636",
        "C_mass": 13,
        "O_A_mass": 16,
        "O_B_mass": 16,
        "is_symmetric": 1,
        "file": "13C-16O2__Dozen.states.cut",
    },
    {
        "id": "727",
        "C_mass": 12,
        "O_A_mass": 17,
        "O_B_mass": 17,
        "is_symmetric": 1,
        "file": "12C-17O2__Dozen.states.cut",
    },
    {
        "id": "737",
        "C_mass": 13,
        "O_A_mass": 17,
        "O_B_mass": 17,
        "is_symmetric": 1,
        "file": "13C-17O2__Dozen.states.cut",
    },
    {
        "id": "828",
        "C_mass": 12,
        "O_A_mass": 18,
        "O_B_mass": 18,
        "is_symmetric": 1,
        "file": "12C-18O2__Dozen.states.cut",
    },
    {
        "id": "838",
        "C_mass": 13,
        "O_A_mass": 18,
        "O_B_mass": 18,
        "is_symmetric": 1,
        "file": "13C-18O2__Dozen.states.cut",
    },
    # Asymmetric isotopologues
    {
        "id": "627",
        "C_mass": 12,
        "O_A_mass": 16,
        "O_B_mass": 17,
        "is_symmetric": 0,
        "file": "16O-12C-17O__Dozen.states.cut",
    },
    {
        "id": "628",
        "C_mass": 12,
        "O_A_mass": 16,
        "O_B_mass": 18,
        "is_symmetric": 0,
        "file": "16O-12C-18O__Dozen.states.cut",
    },
    {
        "id": "637",
        "C_mass": 13,
        "O_A_mass": 16,
        "O_B_mass": 17,
        "is_symmetric": 0,
        "file": "16O-13C-17O__Dozen.states.cut",
    },
    {
        "id": "638",
        "C_mass": 13,
        "O_A_mass": 16,
        "O_B_mass": 18,
        "is_symmetric": 0,
        "file": "16O-13C-18O__Dozen.states.cut",
    },
    {
        "id": "728",
        "C_mass": 12,
        "O_A_mass": 16,
        "O_B_mass": 18,
        "is_symmetric": 0,
        "file": "16O-12C-18O__Dozen.states.cut",
    },
    {
        "id": "738",
        "C_mass": 13,
        "O_A_mass": 16,
        "O_B_mass": 18,
        "is_symmetric": 0,
        "file": "16O-13C-18O__Dozen.states.cut",
    },
]
