import React, { useState } from 'react';
import { Collapse, Button, CardBody, Card } from 'reactstrap';

function Collapsebtn(args) {
    const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <>
      <Button className='p-2 w-100 ' color="secondary" onClick={toggle} style={{ marginBottom: '1rem' }}>
        Profile
      </Button>
      <Collapse isOpen={isOpen} {...args}>
        <Card>
          <CardBody>
            Anim pariatur cliche reprehenderit, enim eiusmod high life accusamus
            terry richardson ad squid. Nihil anim keffiyeh helvetica, craft beer
            labore wes anderson cred nesciunt sapiente ea proident.
          </CardBody>
        </Card>
      </Collapse>
    </>
  );
}

export default Collapsebtn
