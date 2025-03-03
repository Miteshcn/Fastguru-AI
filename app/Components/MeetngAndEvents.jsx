  import React, { useState } from 'react';
import { Collapse, Button, CardBody, Card } from 'reactstrap';

function MeetigsAndEvents(args) {
    const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <>
      <Button className='p-2 w-100 ' color="secondary" onClick={toggle} >
       Meetings & Events
      </Button>
      <Collapse isOpen={isOpen} {...args}>
        <Card>
          <CardBody className='bg-dark flex'>
           
                 <iframe src="https://calendar.google.com/calendar/embed?src=your_calendar_id&ctz=your_timezone" 
               
               >
            </iframe>
             
          </CardBody>
        </Card>
      </Collapse>
    </>
  );
}

export default MeetigsAndEvents




 
 
   
 
