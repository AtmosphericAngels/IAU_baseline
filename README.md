### About

The module is able to classify between baseline and pollution events, depending on the limit set by user. 

### Installation

```
pip install git+https://github.com/thdieder/IAU_pollution_detection_algorithm.git
```

### Usage

`IAU_pollution_detection_algorithm` is easy to use. Import the module:
```
import IAU_pollution_detection_algorithm.pollution_detection as IAU
```
Depending on your data you need to adjust the time values to either datetime format or fractionaltime. <br>
`x_val`, `y_val` and `y_err` need to be lists. 

```
IAU.find_ol(func=IAU.fct.higher, x_val = datetime, y_val = mole_fraction, y_err = stdev_of_molefraction , flag= None)
```

### Example

![Baseline](./example/baseline.png)
