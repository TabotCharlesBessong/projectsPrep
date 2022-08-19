

import React, { useState } from "react";
import { FaDollarSign } from "react-icons/fa";
import FormInputGroup from "../FormInputGroup";
import {Typography} from '@material-ui/core'
import './styles.css'

const Accusition = () => {
	
  const [loanAmount,setLoanAmount] = useState('')
  const [closingCost,setClosingCost] = useState('')
  const [capitalExpenditure,setCapitalExpenditure] = useState('')
  const [acquisitionCost,setAcquisitionCost] = useState('')
  const [monthlyRent,setMonthlyRent] = useState('')
  const [vaccancy,setVaccancy] = useState('')
  const [vaccancyVal, setVaccancyVal] = useState("");
  // const [rentPerUnit,setRentPerUnit] = useState('')
  const [numberOfUnit,setNumberOfUnit] = useState('')
  const [managementCost,setManagementCost] = useState('')
  const [electricBill,setElectricBill] = useState('')
  const [waterBill,setWaterBill] = useState('')
  const [garbage,setGarbage] = useState('')
  const [tax,setTax] = useState('')
  const [insurance,setInsurance] = useState('')
  const [plumbing,setPlumbing] = useState('')
  const [landscape,setLandscape] = useState('')
  const [security,setSecurity] = useState('')
	

  const totalCost = () => {
    const total = Number(loanAmount) + Number(capitalExpenditure) + Number(closingCost)
    setAcquisitionCost(total)
    return acquisitionCost
  }
 
  const totalVaccancy = () => {
    const vac = Number(monthlyRent) * (Number(vaccancy  )  / 100 )
    setVaccancyVal(vac)
    return vaccancyVal
  }

  

	return (
		<>
			<div className="rentalCalculator">
				<form onSubmit={(e) => e.preventDefault()} className="mb-6 mx-auto">
					<FormInputGroup
						text="Loan amount"
						icon={<FaDollarSign />}
						placeholder={"Enter your loan amount"}
						value={loanAmount}
						onInput={(e) => setLoanAmount(e.target.value)}
					/>
					<FormInputGroup
						text="Closing Cost "
						icon={<FaDollarSign />}
						placeholder={"Enter the closing cost"}
						value={closingCost}
						onInput={(e) => setClosingCost(e.target.value)}
					/>
					<FormInputGroup
						text="Cap Expenditure"
						icon={<FaDollarSign />}
						placeholder={"Enter your Capital Expenditure"}
						value={capitalExpenditure}
						onInput={(e) => setCapitalExpenditure(e.target.value)}
					/>

					<Typography variant="h5" gutterBottom>
						total acquisition expenditure : <FaDollarSign />
						<span style={{ marginLeft: "1rem" }}>{acquisitionCost}</span>
					</Typography>

					<button
						type="submit"
						onClick={totalCost}
						className="btn btn-primary btn-md w-50 mb-4 center  ml-5"
					>
						Calculate
					</button>
				</form>

				<form className="mt-6 mx-auto" onSubmit={(e) => e.preventDefault()}>
					<FormInputGroup
						text="Monthly Rent"
						icon={<FaDollarSign />}
						placeholder={"How much is the rent"}
						value={monthlyRent}
						onInput={(e) => setMonthlyRent(e.target.value)}
					/>
					<FormInputGroup
						text="Vaccancy Rate "
						icon="%"
						placeholder={"what is the property vaccancy rate"}
						value={vaccancy}
						onInput={(e) => setVaccancy(e.target.value)}
					/>
					<button
						type="submit"
						onClick={totalVaccancy}
						className="btn btn-primary btn-sm w-50  center mb-4 ml-5 "
					>
						Calculate Vaccancy
					</button>
					<Typography variant="h5" gutterBottom>
						total acquisition expenditure : <FaDollarSign />
						<span style={{ marginLeft: "1rem" }}>{vaccancyVal}</span>
					</Typography>
					<FormInputGroup
						text="Number of unit"
						icon={<FaDollarSign />}
						placeholder={"How many unit does this property have"}
						value={numberOfUnit}
						onInput={(e) => setNumberOfUnit(e.target.value)}
					/>

					<FormInputGroup
						text="Management Cost"
						icon={<FaDollarSign />}
						placeholder={"what is your management cost"}
						value={managementCost}
						onInput={(e) => setManagementCost(e.target.value)}
					/>

					<FormInputGroup
						text="Electric Bill  "
						icon="%"
						placeholder={"electric bill"}
						value={electricBill}
						onInput={(e) => setElectricBill(e.target.value)}
					/>
					<FormInputGroup
						text="Water Bill"
						icon={<FaDollarSign />}
						placeholder={"How much is water"}
						value={waterBill}
						onInput={(e) => setWaterBill(e.target.value)}
					/>

					<FormInputGroup
						text="Garbage Payment"
						icon={<FaDollarSign />}
						placeholder={"How much is garbage cost"}
						value={garbage}
						onInput={(e) => setGarbage(e.target.value)}
					/>
					<FormInputGroup
						text="Property Tax "
						icon={<FaDollarSign />}
						placeholder={"what is the property tax"}
						value={tax}
						onInput={(e) => setTax(e.target.value)}
					/>

					<FormInputGroup
						text="Insurance"
						icon={<FaDollarSign />}
						placeholder={"How much do you pay for insurance"}
						value={insurance}
						onInput={(e) => setInsurance(e.target.value)}
					/>

					<FormInputGroup
						text="Plumbing Bill  "
						icon={<FaDollarSign />}
						placeholder={"Blumbing bill"}
						value={plumbing}
						onInput={(e) => setPlumbing(e.target.value)}
					/>
					<FormInputGroup
						text="Landscape cost"
						icon={<FaDollarSign />}
						placeholder={"what is your landscape cost"}
						value={landscape}
						onInput={(e) => setLandscape(e.target.value)}
					/>

					<FormInputGroup
						text="Tenant security"
						icon={<FaDollarSign />}
						placeholder={"How much is water"}
						value={security}
						onInput={(e) => setSecurity(e.target.value)}
					/>

					<Typography variant="h5" gutterBottom>
						total acquisition expenditure : <FaDollarSign />
						<span style={{ marginLeft: "1rem" }}>{acquisitionCost}</span>
					</Typography>

					<button
						type="submit"
						onClick={totalCost}
						className="btn btn-primary btn-md w-50  center mb-4 ml-5 "
					>
						Calculate
					</button>
				</form>
			</div>
		</>
	);
}

export default Accusition;