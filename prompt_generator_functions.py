### Prompt Generator
# This notebook contains code that will intake numerical data for our planets and output keywords that describe the planet, from size, look, type of star, etc. These keywords will then be put into a single prompt generator to produce a prompt for each planet to go into Stable Diffusion.

import pandas as pd
import numpy as np
import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description="Data Preprocessing for Machine Learning")
    parser.add_argument("--training-data", required=True, help="Path to the training data CSV file")
    parser.add_argument("--exoplanet-data", required=True, help="Path to the exoplanet data CSV file")
    return parser

def preprocess_data(training_data_path, exoplanet_data_path):
    # Load training data and exoplanet data from CSV files
    training_data = pd.read_csv(training_data_path)
    exoplanet_data = pd.read_csv(exoplanet_data_path)

    # Preprocessing the training data
    training_data = training_data.drop('Unnamed: 0', axis=1)
    training_data.fillna(0, inplace=True)

    str_to_float_cols = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_imppar',
                        'st_teff', 'st_rad', 'st_mass', 'sy_vmag']

    for col in str_to_float_cols:
        training_data[col] = training_data[col].astype(str).str.replace(',', '').astype('float32')
    
    training_data['st_spectype'] = training_data['st_spectype'].apply(lambda x: float(x) if x == 0 else x)

    # Preprocessing the exoplanet data
    exoplanet_data = exoplanet_data.drop(0)
    exoplanet_data.fillna(0, inplace=True)

    str_to_float_cols = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_imppar',
                        'st_teff', 'st_rad', 'st_mass', 'sy_vmag']

    exoplanet_data[str_to_float_cols] = exoplanet_data[str_to_float_cols].apply(pd.to_numeric, downcast='float')
    
    return training_data, exoplanet_data

### Getting Planet Information

# The next thing that we want to do is determine what our planet is going to look like. For this, we need to determine 1) it's size, and 2) what type of planet is it (given it's distance from it's star, it's size, temperature, etc.). We'll start with size and go from there. 

# #### Planet Description

# There are different exoplanet types categorized as such:
# * Terrestrial: earth-sized or smaller, mostly made of rock and metal. Half of earth's size to twice earth's size.
# * Super-Earth: typically "terrestrial" or rocky, but more massive than earth and lighter than neptune. They might or might not have atmospheres. Twice to 10 times the mass of earth. 
# * Neptune-like: similar in size to neptune and uranus with hydrogen or helium dominated atmospheres. Mini neptunes are larger than earth, but smaller than neptune. Neptune is about 4 times the size of earth and 17 times as massive as earth. Uranus is 14 times as massive as earth. 
# * Gas Giants: The size of saturn or much larger. They include "hot" jupiters. Jupiter is 11 times larger than earth and 318 times as massive, and saturn is 10 times larger than earth and 95 times more massive. 
# 
# This determines what category the planet falls under. The temperature (or we can calculate that with impact parameter if need be) determines what type of atmosphere the planet may have (if it is a black body). 

# ## Get Planet Category
# 
# In the below section, we are defining a function that will categorize our data into types of planets based on their mass or radius.

def get_planet_category(dataset):
    if dataset['pl_bmasse'].any() != 0.0:
        conditions = [(dataset['pl_bmasse'] > 0.0) & (dataset['pl_bmasse'] <= 2.0),
                    (dataset['pl_bmasse'] > 2.0) & (dataset['pl_bmasse'] <= 10.0),
                    (dataset['pl_bmasse'] > 10.0) & (dataset['pl_bmasse'] <= 17.0),
                    (dataset['pl_bmasse'] > 17.0),
                    (dataset['pl_bmasse'] == 0.0)
                    ]
        
        values = ['terrestrial', 'super-earth', 'neptune-like', 'gas-giant', 'unknown planet size']

        dataset['planet_category'] = np.select(conditions, values, default='unknown')

    elif dataset['pl_bmasse'].any() == 0.0 and dataset['pl_rade'] != 0.0:
        conditions = [(dataset['pl_rade'] > 0.0) & (dataset['pl_rade'] <= 2.0),
                    (dataset['pl_rade'] > 2.0) & (dataset['pl_rade'] <= 10.0),
                    (dataset['pl_rade'] > 10.0) & (dataset['pl_rade'] <= 17.0),
                    (dataset['pl_rade'] > 17.0), 
                    (dataset['pl_rade'] == 0.0)
                    ]
        
        values = ['terrestrial', 'super-earth', 'neptune-like', 'gas-giant', 'unknown planet size']

        dataset['planet_category'] = np.select(conditions, values, default='unknown')
        
    return dataset

# ## Getting Planet Size
# 
#In most variations of our code, we will use the earth mass ratio already within our dataset to determine the size of the planet, however, in one instance of our training, we want to edit this to be not a numerical ratio, but a textual categorization. The below code does this for planets, we will do the same thing later on for our stars.

def planet_mass_description(dataset):
    for index, data in dataset.iterrows():

        mercury_mass = 0.0553
        venus_mass = 0.815
        earth_mass = 1.0
        mars_mass = 0.107
        jupiter_mass = 317.8
        saturn_mass = 95.2
        uranus_mass = 14.5
        neptune_mass = 17.1

        if data['pl_bmasse'] != 0:
            if data['pl_bmasse'] <= mercury_mass:
                dataset.at[index, 'planet_mass_description'] = 'tiny'
            elif mercury_mass < data['pl_bmasse'] <= mars_mass:
                dataset.at[index, 'planet_mass_description'] = 'very small'
            elif mars_mass < data['pl_bmasse'] <= venus_mass:
                dataset.at[index, 'planet_mass_description'] = 'small'
            elif venus_mass < data['pl_bmasse'] <= earth_mass:
                dataset.at[index, 'planet_mass_description'] = 'medium small'
            elif uranus_mass < data['pl_bmasse'] <= neptune_mass:
                dataset.at[index, 'planet_mass_description'] = 'medium'
            elif neptune_mass < data['pl_bmasse'] <= saturn_mass:
                dataset.at[index, 'planet_mass_description'] = 'large'
            elif saturn_mass < data['pl_bmasse'] <= jupiter_mass:
                dataset.at[index, 'planet_mass_description'] = 'giant'
            elif jupiter_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_mass_description'] = 'massive'
                
        elif data['pl_bmasse'] == 0:
            if data['planet_category'] == 'terrestrial':
                dataset.at[index, 'planet_mass_description'] = 'small'
            elif data['planet_category'] == 'super-earth':
                dataset.at[index, 'planet_mass_description'] = 'medium'
            elif data['planet_category'] == 'neptune-like':
                dataset.at[index, 'planet_mass_description'] = 'large'
            elif data['planet_category'] == 'gas-giant':
                dataset.at[index, 'planet_mass_description'] = 'giant'

        else:
            dataset.at[index, 'planet_mass_description'] = 'unknown size'
            
    return dataset

# ## Getting Planet_Color Description
# With the below code, we are defining a function that will return the planet_color based on scientific backed research into the available data in our dataset.

def get_planet_description(dataset):
    for index, data in dataset.iterrows():
        #mass of the planets in our solar system for comparative purposes
        mercury_mass = 0.0553
        venus_mass = 0.815
        earth_mass = 1.0
        mars_mass = 0.107
        jupiter_mass = 317.8
        saturn_mass = 95.2
        uranus_mass = 14.5
        neptune_mass = 17.1

        if data['pl_eqt'] != 0.0: 

        #coding based on type of planet and planet temperature
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth':
                if data['pl_eqt'] <= 20.0:
                    dataset.at[index, 'planet_color'] = 'has a composition of hydrogen and helium producing a distince white color'
                elif 20.0 < data['pl_eqt'] <= 200.0:
                    dataset.at[index, 'planet_color'] = 'has high quantities of methane known for its rich blue color'
                elif 200.0 < data['pl_eqt'] <= 400.0:
                    dataset.at[index, 'planet_color'] = 'likely has a small amount of blue methane and yellow ammonia. The most dominant color would come from blue liquid water'
                elif 400.0 < data['pl_eqt'] <= 600.0:
                    dataset.at[index, 'planet_color'] = 'most likely has water vapor that still produces a true blue color mixing with the breakdown of methanes deep blue'
                elif 600.0 < data['pl_eqt'] <= 800.0:
                    dataset.at[index, 'planet_color'] = 'has carbon dioxide and hydrocarbons are dominant in this planet which could both come in varying shades of blue and white'
                elif 800.0 < data['pl_eqt'] <= 1200.0:
                    dataset.at[index, 'planet_color'] = 'has white carbon dioxide molecules and pale yellow sulfur compounds are likely on this planet'
                elif 1200.0 < data['pl_eqt'] <= 1700.0:
                    dataset.at[index, 'planet_color'] = 'has pale yellow sulfure compounds and blue and white water vapor are likely dominate on this planet'
                elif 1700.0 < data['pl_eqt']:
                    dataset.at[index, 'planet_color'] = 'is so hot all metals are breaking down causing the planet to likely be covered in lava'
                
            elif data['planet_category'] == 'neptune-like':
                if data['pl_eqt'] <= 90.0:
                    dataset.at[index, 'planet_color'] = 'consists mostly of helium and hydrogen which are dominantly white, but it mixes with frozen methane characterized by a light blue color'
                elif 90.0 < data['pl_eqt'] <= 110.0:
                    dataset.at[index, 'planet_color'] = 'has methane as a liquid and dominant in the atmosphere shifting the color to a azure blue color'
                elif 110.0 < data['pl_eqt'] <= 275.0:
                    dataset.at[index, 'planet_color'] = 'has methane as a gas and producing a deep blue color'
                elif 275.0 < data['pl_eqt'] <= 375.0:
                    dataset.at[index, 'planet_color'] = 'has a dark blue methane color mixing with water vapor clouds of a much lighter blue color and traces of ammonia as a light yellow color'
                elif 375.0 < data['pl_eqt'] <= 500.0:
                    dataset.at[index, 'planet_color'] = 'methane is breaking down and possibly mixing with other chemicals such as sulfur, known for its pale yellow color'
                elif 500.0 < data['pl_eqt'] <= 800.0:
                    dataset.at[index, 'planet_color'] = 'methane is breaking down, so the planet is likely no longer a deep blue, but hydrocarbons are likely present in the atmosphere, which depending on composition are varying shades of blue'
                elif 800.0 < data['pl_eqt'] <= 900.0:
                    dataset.at[index, 'planet_color'] = 'has deep blue methane is breaking down and less pronounced and likely to have alkali metals known for their silvery white color'
                elif 900.0 < data['pl_eqt'] <= 1400.0:
                    dataset.at[index, 'planet_color'] = 'has deep blue methane is breaking down and less pronounced, aerosols and thermal emissions are more likely and often give off a neutral or red color that would mix with the blue'
                elif 1400.0 < data['pl_eqt']:
                    dataset.at[index, 'planet_color'] = 'likely overtaken by aerosols and thermal emissions as well as high-temperature gases causing it to be between purple and red in color'
            
            elif data['planet_category'] == 'gas-giant':
                if data['pl_eqt'] <= 70.0:
                    dataset.at[index, 'planet_color'] = 'has frozen ammonia producing a duller yellow color merging with the more dominant methane, characterized by its shade of blue'
                elif 70.0 < data['pl_eqt'] <= 150.0:
                    dataset.at[index, 'planet_color'] = 'most likely overrun with ammonia clouds characterized by their variety of yellow coloring'
                elif 150.0 < data['pl_eqt'] <= 250.0:
                    dataset.at[index, 'planet_color'] = 'has methane in its blue color but in very small quantities. The dominant color will be ammonia, which is now a liquid giving the planet a darker yellow color closer to brown'
                elif 250.0 < data['pl_eqt'] <= 350.0: 
                    dataset.at[index, 'planet_color'] = 'the atmosphere is overtaken with water vapor giving the planet a mostly white color with the posibility of slight blue tinting'
                elif 350.0 < data['pl_eqt'] <= 800.0:
                    dataset.at[index, 'planet_color'] = 'is so warm it likely does not have clouds and appears as a uniform blue orb'
                elif 800.0 < data['pl_eqt'] <= 900.0:
                    dataset.at[index, 'planet_color'] = 'is in transition from a blue atmosphere to being overtaken by carbon monoxide and alkali metals known for being silvery white'
                elif 900.0 < data['pl_eqt']<= 1400.0:
                    dataset.at[index, 'planet_color'] = 'has carbon monoxide and alkali metals like sodium and potassium as dominant, which known for their silvery white coloring'
                elif 1400.0 < data['pl_eqt']:
                    dataset.at[index, 'planet_color'] = 'is dominated by silicate and iron clouds most notably variations of red coloring'
            
            elif data['planet_category'] == 'unknown planet size':
                dataset.at[index, 'planet_color'] = 'is an unknown planet color'
            
        
        elif data['pl_eqt'] == 0.0 and data['pl_bmasse'] != 0.0: 
        #coding based on type of planet and mass
            if data['pl_bmasse'] <= mercury_mass:
                dataset.at[index, 'planet_color'] = 'is likely extremely hot and possibly covered in lava, primary composed of silicate minerals and oxides ranging in a variety of colors from silvery gray to a rich deep red'
            elif mercury_mass < data['pl_bmasse'] <= mars_mass:
                dataset.at[index, 'planet_color'] = 'is primarily composed of silicate minerals and oxides ranging in a variety of colors from silvery gray to a rich deep red'
            elif mars_mass < data['pl_bmasse'] <= venus_mass:
                dataset.at[index, 'planet_color'] = 'is primarily composed of silicate minerals and oxides ranging in a variety of colors from silvery gray to a rich deep red, as well as other gas chemicals such as carbon dioxide which produces a white color, and sulfur known for being a pale yellow'
            elif venus_mass < data['pl_bmasse'] <= earth_mass:
                dataset.at[index, 'planet_color'] = 'likely has a mixture of blue liquid water, and other gas chemicals such as carbon dioxide which produces a white color, and sulfur known for being pale yellow in color'
            elif earth_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_color'] = 'likely has water vapor producing a blue color as well as helium and hydrogen, which both produce shades of white'
            elif data['pl_bmasse'] <= uranus_mass:
                dataset.at[index, 'planet_color'] = 'consisting mostly of helium and hydrogen which are dominantly white, but it mixes with frozen methane characterized by a light blue color'
            elif uranus_mass < data['pl_bmasse'] <= neptune_mass:
                dataset.at[index, 'planet_color'] = 'has methane as a liquid and dominant in the atmosphere shifting the color to a azure blue color'
            elif neptune_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_color'] = 'has methane as a gas and producing a deep blue color'
            elif data['pl_bmasse'] <= saturn_mass:
                dataset.at[index, 'planet_color'] = 'has frozen ammonia producing a duller yellow color with slight traces of methane characterized by its shade of true blue'
            elif saturn_mass < data['pl_bmasse'] <= jupiter_mass:
                dataset.at[index, 'planet_color'] = 'is most likely overrun with ammonia clouds characterized by their variety of yellow coloring'
            elif jupiter_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_color'] = 'has methane in its blue color but in very small quantities with the dominant color will be ammonia, which is now a liquid giving the planet a range of darker yellow and brown colors'

        else:
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth':
                dataset.at[index, 'planet_color'] = 'a rocky world made up of metals and rocks'
            elif data['planet_category'] == 'neptune-like':
                dataset.at[index, 'planet_color'] = 'an icy world composed of frozen gases'
            elif data['planet_category'] == 'gas-giant':
                dataset.at[index, 'planet_color'] = 'a giant world obscured by swirling gases'
            elif data['planet_category'] == 'unknown planet size':
                dataset.at[index, 'planet_color'] = 'has an unknown planet color'
        
    return dataset

### Shortened Description
# 
# Here we are generating a shorter planet_color description to use as an option when testing.
def get_planet_description_short(dataset):
    for index, data in dataset.iterrows():
        #need to code in the mass of all the planets in our solar system
        mercury_mass = 0.0553
        venus_mass = 0.815
        earth_mass = 1.0
        mars_mass = 0.107
        jupiter_mass = 317.8
        saturn_mass = 95.2
        uranus_mass = 14.5
        neptune_mass = 17.1

        if data['pl_eqt'] != 0.0: 

        #coding based on type of planet and planet temperature
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth':
                if data['pl_eqt'] <= 20.0:
                    dataset.at[index, 'planet_color_short'] = 'is white in color'
                elif 20.0 < data['pl_eqt'] <= 200.0:
                    dataset.at[index, 'planet_color_short'] = 'is rich blue in color'
                elif 200.0 < data['pl_eqt'] <= 400.0:
                    dataset.at[index, 'planet_color_short'] = 'contains liquid water, and has traces of blue and yellow coloring'
                elif 400.0 < data['pl_eqt'] <= 600.0:
                    dataset.at[index, 'planet_color_short'] = 'is a shade of blue in color'
                elif 600.0 < data['pl_eqt'] <= 800.0:
                    dataset.at[index, 'planet_color_short'] = 'is a varying shade of blue and/or white'
                elif 800.0 < data['pl_eqt'] <= 1200.0:
                    dataset.at[index, 'planet_color_short'] = 'is white and pale yellow in color'
                elif 1200.0 < data['pl_eqt'] <= 1700.0:
                    dataset.at[index, 'planet_color_short'] = 'is mostly blue and white with possible pale yellow coloring'
                elif 1700.0 < data['pl_eqt']:
                    dataset.at[index, 'planet_color_short'] = 'is covered in lava'
                
            elif data['planet_category'] == 'neptune-like':
                if data['pl_eqt'] <= 90.0:
                    dataset.at[index, 'planet_color_short'] = 'is mostly white mixed with light blue in color'
                elif 90.0 < data['pl_eqt'] <= 110.0:
                    dataset.at[index, 'planet_color_short'] = 'is azure blue in color'
                elif 110.0 < data['pl_eqt'] <= 275.0:
                    dataset.at[index, 'planet_color_short'] = 'is a deep blue color'
                elif 275.0 < data['pl_eqt'] <= 375.0:
                    dataset.at[index, 'planet_color_short'] = 'is mostly a dark blue color mixing with light blue and pale yellow' 
                elif 375.0 < data['pl_eqt'] <= 500.0:
                    dataset.at[index, 'planet_color_short'] = 'is a mixture of blue and yellow in color'
                elif 500.0 < data['pl_eqt'] <= 800.0:
                    dataset.at[index, 'planet_color_short'] = 'is a shade of blue' 
                elif 800.0 < data['pl_eqt'] <= 900.0:
                    dataset.at[index, 'planet_color_short'] = 'is mostly blue mixing with a silvery white color'
                elif 900.0 < data['pl_eqt'] <= 1400.0:
                    dataset.at[index, 'planet_color_short'] = 'is mostly blue mixing with brown and red colors' 
                elif 1400.0 < data['pl_eqt']:
                    dataset.at[index, 'planet_color_short'] = 'is between purple and red in color' 
            
            elif data['planet_category'] == 'gas-giant':
                if data['pl_eqt'] <= 70.0:
                    dataset.at[index, 'planet_color_short'] = 'is pale yellow in color with slight traces of blue'
                elif 70.0 < data['pl_eqt'] <= 150.0:
                    dataset.at[index, 'planet_color_short'] = 'a shade of yellow in color' 
                elif 150.0 < data['pl_eqt'] <= 250.0:
                    dataset.at[index, 'planet_color_short'] = 'a yellow brown in color with slight traces of blue' 
                elif 250.0 < data['pl_eqt'] <= 350.0: 
                    dataset.at[index, 'planet_color_short'] = 'mostly white in color with slight traces of blue' 
                elif 350.0 < data['pl_eqt'] <= 800.0:
                    dataset.at[index, 'planet_color_short'] = 'a uniform blue in color' 
                elif 800.0 < data['pl_eqt'] <= 900.0:
                    dataset.at[index, 'planet_color_short'] = 'is blue mixing with silvery white in color' 
                elif 900.0 < data['pl_eqt']<= 1400.0:
                    dataset.at[index, 'planet_color_short'] = 'is mostly silvery white in color' 
                elif 1400.0 < data['pl_eqt']:
                    dataset.at[index, 'planet_color_short'] = 'a shade of red in color' 
            
            elif data['planet_category'] == 'unknown planet size':
                dataset.at[index, 'planet_color_short'] = 'unknown planet color'
            
        
        elif data['pl_eqt'] == 0.0 and data['pl_bmasse'] != 0.0: 
        #coding based on type of planet and mass
            if data['pl_bmasse'] <= mercury_mass:
                dataset.at[index, 'planet_color_short'] = 'is covered in lava and a shade of deep red to silvery gray in color' 
            elif mercury_mass < data['pl_bmasse'] <= mars_mass:
                dataset.at[index, 'planet_color_short'] = 'a shade of deep red to silvery gray in color'
            elif mars_mass < data['pl_bmasse'] <= venus_mass:
                dataset.at[index, 'planet_color_short'] = 'is likely a shade of deep red to silvery gray with traces of white and pale yellow coloring' 
            elif venus_mass < data['pl_bmasse'] <= earth_mass:
                dataset.at[index, 'planet_color_short'] = 'contains liquid water and possible white and yellow coloring'
            elif earth_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_color_short'] = 'is mostly blue with traces of white coloring' 
            elif data['pl_bmasse'] <= uranus_mass:
                dataset.at[index, 'planet_color_short'] = 'is light blue with traces of white coloring' 
            elif uranus_mass < data['pl_bmasse'] <= neptune_mass:
                dataset.at[index, 'planet_color_short'] = 'an azure blue color'
            elif neptune_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_color_short'] = 'a deep blue color'
            elif data['pl_bmasse'] <= saturn_mass:
                dataset.at[index, 'planet_color_short'] = 'a dull yellow color mixing with true blue'
            elif saturn_mass < data['pl_bmasse'] <= jupiter_mass:
                dataset.at[index, 'planet_color_short'] = 'a shade of yellow coloring'
            elif jupiter_mass < data['pl_bmasse']:
                dataset.at[index, 'planet_color_short'] = 'a darker yellow and brown color with possible blue'

        else:
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth':
                dataset.at[index, 'planet_color_short'] = 'is a rocky world made up of metals and rocks'
            elif data['planet_category'] == 'neptune-like':
                dataset.at[index, 'planet_color_short'] = 'is an icy world composed of frozen gases'
            elif data['planet_category'] == 'gas-giant':
                dataset.at[index, 'planet_color_short'] = 'is a giant world obscured by swirling gases'
            elif data['planet_category'] == 'unknown planet size':
                dataset.at[index, 'planet_color_short'] = 'ia an unknown planet color'
        
    return dataset

# ## Creating a Function to get Orbital Speed
def get_orbital_period(dataset):
    for index, data in dataset.iterrows():
        if data['pl_orbper'] == 0 and data['pl_orbsmax'] !=0:
            orbit_distance = np.sqrt((data.loc['pl_orbsmax'])**3)
            dataset.at[index, 'pl_orbper'] = orbit_distance
    return dataset

# # Creating a function to get planet spin
#adding in a spin column, as the faster a planet spins, the more turbulent it's weather and the more likely it is to have clouds, banding, etc. 
def get_planet_spin(dataset):
    for index, data in dataset.iterrows():
        #planet orbital periods
        mercury_spin = 88
        venus_spin = 224
        earth_spin = 365
        mars_spin = 687
        jupiter_spin = 4332
        saturn_spin = 10747
        uranus_spin = 30589
        neptune_spin = 59800 

        #planet masses 
        mercury_mass = 0.0553
        venus_mass = 0.815
        earth_mass = 1.0
        mars_mass = 0.107
        jupiter_mass = 317.8
        saturn_mass = 95.2
        uranus_mass = 14.5
        neptune_mass = 17.1

        if data['pl_orbper'] != 0.0: #checking if there is an orbital period listed in the dataset
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth': 
                if data['pl_orbper'] <= mercury_spin:
                    dataset.at[index, 'planet_spin'] = 'is hot and rotating quickly with little to no atmosphere, clouds, or storms'
                elif mercury_spin < data['pl_orbper'] <= venus_spin:
                    dataset.at[index, 'planet_spin'] = 'is hot and rotating quickly hot with a thick atmosphere of heavy swirling clouds with bright and dark markings'
                elif venus_spin < data['pl_orbper'] <= earth_spin:
                    dataset.at[index, 'planet_spin'] = 'has clouds of various sizes speckling planet atmosphere showing pieces of the planet terrain beneath'
                elif earth_spin < data['pl_orbper'] <= mars_spin:
                    dataset.at[index, 'planet_spin'] = 'has clouds of various sizes speckling planet atmosphere showing pieces of the planet terrain beneath'
                elif mars_spin < data['pl_orbper']:
                    dataset.at[index, 'planet_spin'] = 'has wisps of clouds of various sizes speckling planet atmosphere showing most of the planet terrain beneath'
            elif data['planet_category'] == 'neptune-like':
                if data['pl_orbper'] <= uranus_spin:
                    dataset.at[index, 'planet_spin'] = 'has clearly defined striped light and dark icy clouds'
                elif uranus_spin < data['pl_orbper'] <= neptune_spin:
                    dataset.at[index, 'planet_spin'] = 'has softly defined striped light and dark icy clouds'
                elif neptune_spin < data['pl_orbper']:
                    dataset.at[index, 'planet_spin'] = 'has icy clouds with no apparent delineation between colors'
            elif data['planet_category'] == 'gas-giant':
                if data['pl_orbper'] <= jupiter_spin:
                    dataset.at[index, 'planet_spin'] = 'has stripes of thick clouds of various coloring defined by clear, sharp edges'
                elif jupiter_spin < data['pl_orbper'] <= saturn_spin:
                    dataset.at[index, 'planet_spin'] = 'has stripes of thick clouds of various coloring defined by softened edges'
                elif saturn_spin < data['pl_orbper']:
                    dataset.at[index, 'planet_spin'] = 'has thick clouds of various coloring blending together across the planet surface'
        elif data['pl_orbper'] == 0.0:  #if zero go off planet mass and compare to the planets in our solar system
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth': 
                if data['pl_bmasse'] <= mercury_mass:
                    dataset.at[index, 'planet_spin'] = 'is hot and rotating quickly with little to no atmosphere, clouds, or storms'
                elif mercury_mass < data['pl_bmasse'] <= venus_mass:
                    dataset.at[index, 'planet_spin'] = 'is hot and rotating quickly with a thick atmosphere of heavy swirling clouds with bright and dark markings'
                elif venus_mass < data['pl_bmasse'] <= earth_mass:
                    dataset.at[index, 'planet_spin'] = 'has clouds of various sizes speckling planet atmosphere showing pieces of the planet terrain beneath'
                elif earth_mass < data['pl_bmasse'] <= mars_mass:
                    dataset.at[index, 'planet_spin'] = 'has clouds of various sizes speckling planet atmosphere showing pieces of the planet terrain beneath'
                elif mars_mass < data['pl_bmasse']:
                    dataset.at[index, 'planet_spin'] = 'has wisps of clouds of various sizes speckling planet atmosphere showing most of the planet terrain beneath'
            elif data['planet_category'] == 'neptune-like':
                if data['pl_bmasse'] <= uranus_mass:
                    dataset.at[index, 'planet_spin'] = 'has clearly defined striped light and dark icy clouds'
                elif uranus_mass < data['pl_bmasse'] <= neptune_mass:
                    dataset.at[index, 'planet_spin'] = 'has softly defined striped light and dark icy clouds'
                elif neptune_mass < data['pl_bmasse']:
                    dataset.at[index, 'planet_spin'] = 'has icy clouds with no apparent delineation between colors'
            elif data['planet_category'] == 'gas-giant':
                if data['pl_bmasse'] <= jupiter_mass:
                    dataset.at[index, 'planet_spin'] = 'has stripes of thick clouds of various coloring defined by clear, sharp edges'
                elif jupiter_mass < data['pl_bmasse'] <= saturn_mass:
                    dataset.at[index, 'planet_spin'] = 'has stripes of thick clouds of various coloring defined by softened edges'
                elif saturn_mass < data['pl_bmasse']:
                    dataset.at[index, 'planet_spin'] = 'has thick clouds of various coloring blending together across the planet surface'
            else:
                dataset.at[index, 'planet_spin'] = 'rotates around its star'

    return dataset
    
#adding in a spin column, as the faster a planet spins, the more turbulent it's weather and the more likely it is to have clouds, banding, etc. 
#this is the shortened description for the shorter prompt
def get_planet_spin_short(dataset):
    for index, data in dataset.iterrows():
        #planet orbit periods
        mercury_spin = 88
        venus_spin = 224
        earth_spin = 365
        mars_spin = 687
        jupiter_spin = 4332
        saturn_spin = 10747
        uranus_spin = 30589
        neptune_spin = 59800 

        #planet masses
        mercury_mass = 0.0553
        venus_mass = 0.815
        earth_mass = 1.0
        mars_mass = 0.107
        jupiter_mass = 317.8
        saturn_mass = 95.2
        uranus_mass = 14.5
        neptune_mass = 17.1

        if data['pl_orbper'] != 0.0: #checking if there is an orbital period listed in the dataset
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth': 
                if data['pl_orbper'] <= mercury_spin:
                    dataset.at[index, 'planet_spin_short'] = 'is hot and rotating quickly with little to no clouds'
                elif mercury_spin < data['pl_orbper'] <= venus_spin:
                    dataset.at[index, 'planet_spin_short'] = 'is hot and rotating quickly hot with swirling clouds of light and dark markings'
                elif venus_spin < data['pl_orbper'] <= earth_spin:
                    dataset.at[index, 'planet_spin_short'] = 'has clouds of various sizes'
                elif earth_spin < data['pl_orbper'] <= mars_spin:
                    dataset.at[index, 'planet_spin_short'] = 'has clouds of various sizes'
                elif mars_spin < data['pl_orbper']:
                    dataset.at[index, 'planet_spin_short'] = 'has thin clouds of various sizes'
            elif data['planet_category'] == 'neptune-like':
                if data['pl_orbper'] <= uranus_spin:
                    dataset.at[index, 'planet_spin_short'] = 'has clearly defined striped light and dark clouds'
                elif uranus_spin < data['pl_orbper'] <= neptune_spin:
                    dataset.at[index, 'planet_spin_short'] = 'has softly defined striped light and dark clouds'
                elif neptune_spin < data['pl_orbper']:
                    dataset.at[index, 'planet_spin_short'] = 'has cloud colors blending together'
            elif data['planet_category'] == 'gas-giant':
                if data['pl_orbper'] <= jupiter_spin:
                    dataset.at[index, 'planet_spin_short'] = 'has clear, sharp-edge stripes of thick clouds'
                elif jupiter_spin < data['pl_orbper'] <= saturn_spin:
                    dataset.at[index, 'planet_spin_short'] = 'has soft-edged stripes of thick clouds'
                elif saturn_spin < data['pl_orbper']:
                    dataset.at[index, 'planet_spin_short'] = 'has thick clouds of various coloring blending together'
        elif data['pl_orbper'] == 0.0:  #if zero go off planet mass and compare to the planets in our solar system
            if data['planet_category'] == 'terrestrial' or data['planet_category'] == 'super-earth': 
                if data['pl_bmasse'] <= mercury_mass:
                    dataset.at[index, 'planet_spin_short'] = 'is hot and rotating quickly with little to no clouds'
                elif mercury_mass < data['pl_bmasse'] <= venus_mass:
                    dataset.at[index, 'planet_spin_short'] = 'is hot and rotating quickly hot with swirling clouds of light and dark markings'
                elif venus_mass < data['pl_bmasse'] <= earth_mass:
                    dataset.at[index, 'planet_spin_short'] = 'has thick clouds of various sizes'
                elif earth_mass < data['pl_bmasse'] <= mars_mass:
                    dataset.at[index, 'planet_spin_short'] = 'has thick clouds of various sizes'
                elif mars_mass < data['pl_bmasse']:
                    dataset.at[index, 'planet_spin_short'] = 'has thin clouds of various sizes'
            elif data['planet_category'] == 'neptune-like':
                if data['pl_bmasse'] <= uranus_mass:
                    dataset.at[index, 'planet_spin_short'] = 'has clearly defined striped light and dark clouds'
                elif uranus_mass < data['pl_bmasse'] <= neptune_mass:
                    dataset.at[index, 'planet_spin_short'] = 'has softly defined striped light and dark icy clouds'
                elif neptune_mass < data['pl_bmasse']:
                    dataset.at[index, 'planet_spin_short'] = 'has cloud colors blending together'
            elif data['planet_category'] == 'gas-giant':
                if data['pl_bmasse'] <= jupiter_mass:
                    dataset.at[index, 'planet_spin_short'] = 'has clear, sharp-edge stripes of thick clouds'
                elif jupiter_mass < data['pl_bmasse'] <= saturn_mass:
                    dataset.at[index, 'planet_spin_short'] = 'has soft-edged stripes of thick clouds'
                elif saturn_mass < data['pl_bmasse']:
                    dataset.at[index, 'planet_spin_short'] = 'has thick clouds of various coloring blending together'
            else:
                dataset.at[index, 'planet_spin_short'] = 'rotates around its star'

    return dataset

# ## Star and Planet Size as a Ratio
# 
# One way we want to define stellar and planet size (to be used for one of the four prompts) is as a ratio between the two. The below code defines this. 

def calculate_stellar_planet_ratio(dataset):
    dataset['stellar_planet_ratio'] = dataset.apply(lambda row:
        ((row.st_mass / (row.pl_bmasse)) * 100)
        if row.st_mass != 0 and row.pl_bmasse != 0 else 0, axis=1)
    return dataset

# ## Tidal Locked Planets
# 
# The below code predicts whether a planet is tidal locked. Tidally locked planets happen when eccentricity is close to zero, it is usually calculated when the rotation of a planet is the same as the orbital period of the planet, however, we do not have enough information to determine the rotation of a planet (you need the angular velocity which we do not have). Another way of doing it is by calculating the roche limit and determining if the imppact parameter is less than that. If it is, it is likely the planet is tidal locked. The below code determines the planet's roche limit and compares it to the pl_imppar to determine whether the planet is tidal locked or not. If the planet does not have the information to determine the roche limit or the pl_imppar, this field is ignored in the prompt generation. 

#getting the roche limit so we can determine if a planet is tidal locked or not

def calculate_roche_limit(dataset):
    return (1.26 * dataset['pl_rade'] * (dataset['st_mass'] / dataset['pl_bmasse']) ** (1/3))\
        .where((dataset['st_mass'] != 0) & (dataset['pl_bmasse'] != 0), 0)

#determining if a planet is tidally locked or not
def tidal_locking(dataset):
    for index, data in dataset.iterrows():
        if data['pl_imppar'] != 0.0 or data['pl_orbsmax'] != 0.0:
            if data['pl_imppar'] or data['pl_orbsmax'] < data['roche_limit']:
                dataset.at[index, 'tidal_locked'] = 'only has one side of the planet facing the sun. The side facing the sun is extremely hot and the side that faces away from the sun is dark and cold'
            
            elif data['pl_imppar'] or data['pl_orbsmax'] >= data['roche_limit']:
                dataset.at[index, 'tidal_locked'] = 'spins around its orbit so both sides get heat from the sun'
            
        else:
            dataset.at[index, 'tidal_locked'] = 0.0

    return dataset

# ## Getting Star Information

# We know from above that there are 3,540 missing spectral types in our dataset

# We can determine star type by a few values. Temperature, mass, and magnitude, however are the best ways.
# We'll be using the Harvard Spectral Classification, which classifies stars based on temperature.
# Stellar classifications include:
# * M - temperature: <3500, color: orange red - light orange red 
# * K - temperature: 3500-5000, color: light orange - pale yellow
# * G - temperature: 5000-6000,  yellow - yellow-white
# * F - temperature: 6000-7500, yellow-white - white
# * A - temperature: 7500-11000 - white - blue-white
# * B - temperature: 11000 - 25000 - blue-white - deep-blue-white
# * O - temperature: 25000-100000 - blue
# * White-Dwarf - temperature: >100000, color: white
# 
# Note this is just giving us color! We will use star mass to then determine size. 
# 

# #### Stellar Color

def get_stellar_color(dataset):
    for index, data in dataset.iterrows():
        if data['st_spectype'] != 0:
            #harvard standard spectral classifications
            if data['st_spectype'][0] == 'M' or data['st_spectype'][0] == 'm':
                dataset.at[index, 'stellar_color'] = 'orange red'
            elif data['st_spectype'][0] == 'K':
                dataset.at[index, 'stellar_color'] = 'light orange'
            elif data['st_spectype'][0] == 'G':
                dataset.at[index, 'stellar_color'] = 'yellow'
            elif data['st_spectype'][0] == 'F':
                dataset.at[index, 'stellar_color'] = 'yellow white'
            elif data['st_spectype'][0] == 'A':
                dataset.at[index, 'stellar_color'] = 'white'
            elif data['st_spectype'][0] == 'B':
                dataset.at[index, 'stellar_color'] = 'blue white'
            elif data['st_spectype'][0] == 'O':
                dataset.at[index, 'stellar_color'] = 'blue'
            #special case star classifications 
            elif data['st_spectype'][0] == 'T': #late stage brown dwarf 
                dataset.at[index, 'stellar_color'] = 'violet'
            elif data['st_spectype'][0] == 'L': #early stage brown dwarf 
                dataset.at[index, 'stellar_color'] = 'magenta'
            elif data['st_spectype'] == 'WD' or data['st_spectype'][0] == 'D': #white dwarf
                dataset.at[index, 'stellar_color'] = 'white'
            elif data['st_spectype'][0] == 's':
                if data['st_teff'] <= 3500.0:
                    dataset.at[index, 'stellar_color'] = 'orange red'
                elif 3500.0 < data['st_teff'] <= 5000.0:
                    dataset.at[index, 'stellar_color'] = 'light orange'
                elif 5000.0 < data['st_teff'] <= 6000.0:
                    dataset.at[index, 'stellar_color'] = 'yellow'
                elif  6000.0 < data['st_teff'] <= 7500.0:
                    dataset.at[index, 'stellar_color'] = 'yellow white'
                elif 7500.0 < data['st_teff'] <= 11000.0:
                    dataset.at[index, 'stellar_color'] = 'white'
                elif 11000.0 < data['st_teff'] <= 25000.0:
                    dataset.at[index, 'stellar_color'] = 'blue white'
                elif 25000.0 < data['st_teff'] <= 100000.0:
                    dataset.at[index, 'stellar_color'] = 'blue'
                elif 100000.0 < data['st_teff']:
                    dataset.at[index, 'stellar_color'] = 'white'
                       
        elif data['st_spectype'] == 0:
            #print(dataset.at[index, 'st_spectype'])
            if data['st_teff'] <= 3500.0:
                dataset.at[index, 'stellar_color'] = 'orange red'
            elif 3500.0 < data['st_teff'] <= 5000.0:
                dataset.at[index, 'stellar_color'] = 'light orange'
            elif 5000.0 < data['st_teff'] <= 6000.0:
                dataset.at[index, 'stellar_color'] = 'yellow'
            elif  6000.0 < data['st_teff'] <= 7500.0:
                dataset.at[index, 'stellar_color'] = 'yellow white'
            elif 7500.0 < data['st_teff'] <= 11000.0:
                dataset.at[index, 'stellar_color'] = 'white'
            elif 11000.0 < data['st_teff'] <= 25000.0:
                dataset.at[index, 'stellar_color'] = 'blue white'
            elif 25000.0 < data['st_teff'] <= 100000.0:
                dataset.at[index, 'stellar_color'] = 'blue'
            elif 100000.0 < data['st_teff']:
                dataset.at[index, 'stellar_color'] = 'white'

        elif data['st_spectype'] == 0 & data['st_teff'] == 0:
            #print(dataset.at[index, 'st_spectype'])
            if data['st_mass'] <= 0.45:
                dataset.at[index, 'stellar_color'] = 'orange red'
            elif 0.45 < data['st_mass'] <= 0.8:
                dataset.at[index, 'stellar_color'] = 'light orange'
            elif 0.8 < data['st_mass'] <= 1.04:
                dataset.at[index, 'stellar_color'] = 'yellow'
            elif  1.04 < data['st_mass'] <= 1.4:
                dataset.at[index, 'stellar_color'] = 'yellow white'
            elif 1.4 < data['st_mass'] <= 2.1:
                dataset.at[index, 'stellar_color'] = 'white'
            elif 2.1 < data['st_mass'] <= 16:
                dataset.at[index, 'stellar_color'] = 'blue white'
            elif 16 < data['st_mass']:
                dataset.at[index, 'stellar_color'] = 'blue'

        else:
            dataset.at[index, 'stellar_color'] = 'unknown stellar color'
            
    return dataset

# #### Stellar Size: As a Description

# Now that we have stellar color, we need to get descriptive words for stellar size. The stellar sizes are typically associated with their classification, which includes things like mass, temperature, radius, color. The easiest thing for us to do, is actually to use stellar classification again. We'll actually want to use it when we run through our function above. This mean's we're going to have to hard code in all of the different types of stars in our dataset and make special adjustments for the special cases like white dwarfs, red giants, etc. 
# 
# When we go through star sizes, all of the stellar sizes are associated with a particular classification and are also generalized by temperature (so also color) and mass. The most accurate way, however, is mass. We're going to use three different types of mass classifications. One, is going to use the mass value (which is quantified by it's relationship to the size of our sun) as a percentage. Second, is translating these size categories to adjectives. 
# 
# For the first option, we are going to use the st_mass defined already in our dataset. This does not require any additional code.
# 
# For the second option, we are going to create a new column at the same time as creating our stellar_color column above. We will associate the following key words to each type of star: 
# 
# D/WD (white dwarf) = tiny
# T/L (brown dwarf) = tiny
# s (sub dwarf) = tiny
# M = very small
# K = small
# G = medium small
# F = medium
# A = large
# B = giant
# O = massive
# 
# For the third option, we are going to use the same ratio between planet and star size defined above. These three are mirrored by the planet's size categories and will be put together for the prompt generation. 
#

def stellar_mass_description(dataset):
    for index, data in dataset.iterrows():
        st_spectype = str(data['st_spectype'])
        if st_spectype[0] == 'W' or st_spectype[0] == 'D' or st_spectype[0] == 'L' or st_spectype[0] == 'T' or st_spectype[0] == 's':
            dataset.at[index, 'stellar_mass_description'] = 'tiny'
        elif data['stellar_color'] == 'orange red':
            dataset.at[index, 'stellar_mass_description'] = 'very small'
        elif data['stellar_color'] == 'light orange':
            dataset.at[index, 'stellar_mass_description'] = 'small'
        elif data['stellar_color'] == 'yellow':
            dataset.at[index, 'stellar_mass_description'] = 'medium small'
        elif data['stellar_color'] == 'yellow white':
            dataset.at[index, 'stellar_mass_description'] = 'medium'
        elif data['stellar_color'] == 'white':
            dataset.at[index, 'stellar_mass_description'] = 'large'
        elif data['stellar_color'] == 'blue white':
            dataset.at[index, 'stellar_mass_description'] = 'giant'
        elif data['stellar_color'] == 'blue':
            dataset.at[index, 'stellar_mass_description'] = 'massive'
        else:
            dataset.at[index, 'stellar_mass_description'] = 'unknown size'
    return dataset

# ### Creating the Image Prompt

# When we make our description we need to make sure we are not only listing the star and the planet descriptions, but we also want to list if there are any other stars in the solar system, any other planets, and if there are any moons. 

# The foundation format:
# 
# "A solar system made up of {sy_pnum} planet(s), {sy_snum} star(s), and {sy_mnum} moon(s). This {planet_category} planet is {pl_bmasse} the size of earth, {planet_color}, and {planet_spin}. This planet {tidal_locked}. The planet\'s star is {stellar_color} and {st_mass} the size of our sun."

def get_prompts(dataset):
    for index, data in dataset.iterrows():

        pl_bmasse_int = int(data['pl_bmasse']) #realized it looked kinda messy as floats, so we're making them integers before running them through our prompt generator
        #also means we are getting rid of this f"{data['pl_bmasse']} times" if data['pl_bmasse'] != 0 else "an unknown size compared to" and this f"{data['st_mass']} times" if 'st_mass' != 0 else "an unknown size compared to"
        st_mass_int = int(data['st_mass'])

        #creating a prompt with star size and planet size as numbers (our foundation prompt)
        dataset.at[index, 'mass_prompt'] = "A solar system made up of {} planet(s), {} star(s), and {} moon(s). This {} planet is {} the size of earth, {} and {}. This planet {}. The planet\'s star is {} and {} the size of the sun.".format(
            data['sy_pnum'], data['sy_snum'], data['sy_mnum'], data['planet_category'], f"{pl_bmasse_int} times" if pl_bmasse_int != 0 else "an unknown size compared to", data['planet_color'], data['planet_spin'], f"{data['tidal_locked']}" if data['tidal_locked'] != 0 else 'has an unknown spin', data['stellar_color'], f"{st_mass_int} times" if st_mass_int != 0 else "an unknown size compared to")

        #creating a prompt with star size and planet size as a ratio
        dataset.at[index, 'ratio_prompt'] = "A solar system made up of {} planet(s), {} star(s), and {} moon(s). This {} planet is {} the size of earth, {} and {}. This planet {}. The planet\'s star is {} and {} the size of it\'s planet.".format(
            data['sy_pnum'], data['sy_snum'], data['sy_mnum'], data['planet_category'], f"{pl_bmasse_int} times" if pl_bmasse_int != 0 else "an unknown size compared to", data['planet_color'], data['planet_spin'], f"{data['tidal_locked']}" if data['tidal_locked'] != 0 else 'has an unknown spin', data['stellar_color'], data['stellar_planet_ratio'])

        #creating a prompt with star and planet size as text
        dataset.at[index, 'size_text_prompt'] = "A solar system made up of {} planet(s), {} star(s), and {} moon(s). This {} {} planet {}. This planet {}. The planet\'s star is {} and {}.".format(
            data['sy_pnum'], data['sy_snum'], data['sy_mnum'], data['planet_mass_description'], data['planet_category'], data['planet_color'], data['planet_spin'], f"{data['tidal_locked']}" if data['tidal_locked'] != 0 else 'has an unknown spin', data['stellar_color'], data['stellar_mass_description'])
        
        #creating a prompt with 75 tokens
        dataset.at[index, '75_tokens'] = 'A {}, {} star with a {}, {} planet. The planet {}, {}, and {}'.format(data['stellar_color'], data['stellar_mass_description'], data['planet_mass_description'], data['planet_category'], data['planet_color_short'], data['planet_spin_short'], data['tidal_locked'])

    return dataset

def save_datasets(exoplanet_data, training_data):
    exoplanet_data.to_csv('exoplanet_data_prompts.csv', index=False)
    training_data.to_csv('training_data_prompts.csv', index=False)

def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # Load training data and exoplanet data from CSV files
    training_data = pd.read_csv(args.training_data)
    exoplanet_data = pd.read_csv(args.exoplanet_data)

    # Preprocess the data using the common preprocessing functions
    training_data = preprocess_data(training_data)
    exoplanet_data = preprocess_data(exoplanet_data)

    # get planet category
    training_data = get_planet_category(training_data)
    exoplanet_data = get_planet_category(exoplanet_data)
    
    # get planet mass description
    training_data = planet_mass_description(training_data)
    exoplanet_data = planet_mass_description(exoplanet_data)
    
    #get planet color
    training_data = planet_description(training_data)
    exoplanet_data = planet_description(exoplanet_data)
    
    # get planet description - short
    training_data = planet_description_short(training_data)
    exoplanet_data = planet_description_short(exoplanet_data)
    
    # get orbital period
    training_data = get_orbital_period(training_data)
    exoplanet_data = get_orbital_period(exoplanet_data)
    
    # get planet_spin
    training_data = get_planet_spin(training_data)
    exoplanet_data = get_planet_spin(exoplanet_data)
    
    # get planet_spin - short
    training_data = get_planet_spin_short(training_data)
    exoplanet_data = get_planet_spin_short(exoplanet_data)
    
    #get tidal locking
    training_data = tidal_locked(training_data)
    exoplanet_data = tidal_locked(exoplanet_data)
    
    # Calculate the stellar-planet ratio for both datasets
    exoplanet_data = calculate_stellar_planet_ratio(exoplanet_data)
    training_data = calculate_stellar_planet_ratio(training_data)
    
    #get stellar color
    training_data = get_stellar_color(training_data)
    exoplanet_data = get_stellar_color(exoplanet_data)
    
    #get stellar mass description
    training_data = stellar_mass_description(training_data)
    exoplanet_data = stellar_mass_description(exoplanet_data)
    
    # Calculate Roche limit for both datasets
    exoplanet_data['roche_limit'] = calculate_roche_limit(exoplanet_data)
    training_data['roche_limit'] = calculate_roche_limit(training_data)
    
    # Generate prompts for both datasets
    exoplanet_data = get_prompts(exoplanet_data)
    training_data = get_prompts(training_data)
    
    save_datasets(exoplanet_data, training_data)

if __name__ == "__main__":
    main()
