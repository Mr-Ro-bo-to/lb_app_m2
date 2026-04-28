

import pint

ureg = pint.get_application_registry()

# ureg.define('[arbitrary] = []')
# ureg.define('arbitrary_unit = [arbitrary] = arb')

# Change default format
ureg.default_format = '~P'  # Compact pretty format
