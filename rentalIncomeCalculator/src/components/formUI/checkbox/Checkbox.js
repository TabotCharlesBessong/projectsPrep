
import React from 'react'
import {Checkbox , FormControl , FormControlLabel,FormLabel,FormGroup,Typography} from '@material-ui/core'

import { useField  , useFormikContext} from 'formik'

const CheckBox = ({
  name,
  label,
  legend,
  options,
  ...otherProps
}) => {
  
  const {setFieldValue} = useFormikContext()
  const [field,meta] = useField(name)

  const handleChange = (e)=>{
    const {checked} = e.target 
    setFieldValue(name,checked)
  }

  const configCheckbox = {
    ...field,
    onChange:handleChange
  }

  const configFormControl = {}
  if(meta && meta.touched && meta.error){
    configFormControl.error = true
  }
  return (
		<FormControl>
			<FormLabel color='red'  component="legend">{legend}</FormLabel>
			<FormGroup
				style={{
					color: "#00e676",
				}}
			>
				{Object.keys(options).map((item, pos) => {
					return (
						<FormControlLabel
							color="red"
							key={pos}
							style={{
								width: "100%",
							}}
							control={<Checkbox {...configCheckbox} />}
							label={options[item]}

							// label={
							// 	<Typography variant="h6" style={{ color: "#2979ff" }}>
							// 		{label}
							// 	</Typography>
							// }
						/>
					);
				})}
			</FormGroup>
		</FormControl>
	);
}

export default CheckBox