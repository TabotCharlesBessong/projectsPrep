

import React from 'react'
import { Radio,RadioGroup,FormControl,FormLabel , FormControlLabel  } from '@material-ui/core'
import {  useField , useFormikContext } from 'formik'


const RadioGroups = ({
    name,
    label,
    legend,
    options,
    direction,
    ...otherProps
}) => {
  const {setFieldValue} = useFormikContext()
  const [field,meta] = useField(name)

  const handleChange = (e)=>{
    const {value} = e.target 
    setFieldValue(name,value)
  }

  const configRadio = {
    ...field,
    ...otherProps,
    radio:true,
    variant:'outlined',
    fullWidth:true,
    onChange:handleChange
  }

  if(meta && meta.touched && meta.error){
    configRadio.error = true
    configRadio.helperText = meta.error
  }

  return (
    <FormControl>
      <FormLabel component="legend">{legend}</FormLabel>
      <RadioGroup style={{display: 'flex', flexDirection: ''}} >
        {Object.keys(options).map((item,pos)=> {
          return(
          <FormControlLabel key={pos} value={item} control={<Radio />} label={options[item]} />
         )
        })}
      </RadioGroup>
    </FormControl>
  )
}


export default RadioGroups

