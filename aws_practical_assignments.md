# AWS Practical Assignments and Assessments
## Case-Driven Lab Tasks for AWS Specialization Topics

---

## Topic 1: Cloud Introduction, VPC Design, Transit Gateway, PrivateLink

### **Practical Assignment 1: Multi-VPC Enterprise Network Architecture**

**Case Study:**
TechCorp is expanding from a single office to 5 global locations (US East, US West, Europe, Asia, Australia). They need a scalable network architecture that supports 3 environments (Production, Staging, Development) across all regions. The architecture must allow controlled communication between environments, centralized internet access, and secure connectivity to on-premises data centers.

**Tasks to Complete:**
1. Design and implement a hub-and-spoke VPC architecture using Transit Gateway
2. Create VPCs for each environment in 2 regions (6 VPCs total)
3. Configure Transit Gateway with route tables for environment isolation
4. Implement centralized internet access through a shared services VPC
5. Set up VPC Flow Logs for network monitoring
6. Configure Route 53 private hosted zones for internal DNS resolution
7. Create security groups and NACLs following principle of least privilege
8. Document network topology and routing decisions

**Assessment Criteria:**
- Network segmentation and security (25%)
- Routing configuration and optimization (25%)
- Documentation and architecture diagrams (20%)
- Cost optimization considerations (15%)
- Scalability and maintenance (15%)

**Deliverables:**
- CloudFormation templates for infrastructure
- Network architecture diagram
- Security analysis report
- Cost estimation document

---

### **Practical Assignment 2: Hybrid Cloud Connectivity with Direct Connect**

**Case Study:**
FinanceSecure Corp operates critical trading applications that require consistent, low-latency connectivity between their on-premises trading floor and AWS. They need 99.99% availability, sub-10ms latency, and dedicated bandwidth of 10Gbps. The solution must include backup connectivity and support for multiple VPCs across different AWS accounts.

**Tasks to Complete:**
1. Design Direct Connect architecture with redundancy
2. Implement Virtual Interfaces (VIFs) for multiple VPCs
3. Configure BGP routing with path preference
4. Set up backup Site-to-Site VPN connectivity
5. Implement PrivateLink for secure service access
6. Create monitoring and alerting for connection health
7. Test failover scenarios and document RTO/RPO
8. Optimize routing for latency-sensitive applications

**Assessment Criteria:**
- High availability design (30%)
- Network performance optimization (25%)
- Failover testing and documentation (20%)
- Cost vs. performance analysis (15%)
- Security implementation (10%)

**Deliverables:**
- Infrastructure as Code templates
- Failover testing report
- Performance benchmark results
- Disaster recovery documentation

---

## Topic 2: EC2, Auto Scaling, Elastic Load Balancing

### **Practical Assignment 1: Auto-Scaling Web Application Platform**

**Case Study:**
NewsDaily operates a breaking news website that experiences unpredictable traffic spikes. Normal traffic is 1,000 concurrent users, but during major events, it can spike to 100,000 users within 15 minutes. The application must maintain response times under 2 seconds and achieve 99.9% availability while optimizing costs during low-traffic periods.

**Tasks to Complete:**
1. Deploy a multi-tier web application across 3 AZs
2. Configure Application Load Balancer with health checks
3. Implement Auto Scaling with predictive and dynamic scaling
4. Set up CloudWatch metrics and custom alarms
5. Configure target tracking scaling policies
6. Implement blue-green deployment strategy
7. Create stress testing scenarios to validate scaling
8. Optimize instance types and pricing models

**Assessment Criteria:**
- Scaling effectiveness and speed (30%)
- High availability implementation (25%)
- Performance under load (20%)
- Cost optimization (15%)
- Monitoring and alerting (10%)

**Deliverables:**
- Auto Scaling configuration documentation
- Load testing results and analysis
- Cost optimization report
- Incident response procedures

---

### **Practical Assignment 2: Container Orchestration with ECS/EKS**

**Case Study:**
MicroservicesCorp is migrating from a monolithic application to microservices architecture. They have 12 microservices that need to scale independently, communicate securely, and support rolling updates with zero downtime. The solution must handle service discovery, configuration management, and observability across all services.

**Tasks to Complete:**
1. Containerize applications using Docker
2. Deploy microservices using ECS with Fargate
3. Implement service discovery and load balancing
4. Configure auto-scaling for individual services
5. Set up centralized logging with CloudWatch
6. Implement distributed tracing with X-Ray
7. Create CI/CD pipeline for container deployments
8. Configure service mesh for inter-service communication

**Assessment Criteria:**
- Microservices architecture design (25%)
- Container orchestration implementation (25%)
- Service communication and discovery (20%)
- Observability and monitoring (15%)
- CI/CD integration (15%)

**Deliverables:**
- Container images and deployment manifests
- Service architecture documentation
- Monitoring dashboard configurations
- CI/CD pipeline documentation

---

## Topic 3: AWS Security Hub, IAM, GuardDuty

### **Practical Assignment 1: Multi-Account Security Governance**

**Case Study:**
SecureEnterprise manages 50 AWS accounts across different business units. They need centralized security monitoring, compliance reporting against SOC 2 and PCI DSS standards, and automated incident response. The solution must provide role-based access control, detect security threats in real-time, and maintain audit trails for all activities.

**Tasks to Complete:**
1. Configure AWS Organizations with SCPs for security controls
2. Implement AWS Security Hub across all accounts
3. Enable GuardDuty with custom threat intelligence
4. Set up centralized CloudTrail logging
5. Configure AWS Config for compliance monitoring
6. Implement automated remediation with Lambda
7. Create security dashboards and reporting
8. Develop incident response workflows

**Assessment Criteria:**
- Multi-account security architecture (30%)
- Threat detection and response (25%)
- Compliance monitoring (20%)
- Automation implementation (15%)
- Documentation and procedures (10%)

**Deliverables:**
- Security architecture documentation
- Automated remediation scripts
- Compliance assessment reports
- Incident response playbooks

---

### **Practical Assignment 2: Zero Trust Network Security**

**Case Study:**
CloudFirst Corp wants to implement a zero trust security model for their cloud infrastructure. Employees work remotely and access applications hosted on AWS. The solution must verify every access request, implement least privilege access, and provide seamless user experience while maintaining strong security posture.

**Tasks to Complete:**
1. Implement AWS SSO with multi-factor authentication
2. Configure fine-grained IAM policies with permission boundaries
3. Set up VPC endpoints for private service access
4. Deploy AWS WAF with custom rules and rate limiting
5. Configure AWS Macie for data classification and protection
6. Implement AWS Inspector for vulnerability assessment
7. Create identity-based and resource-based policies
8. Set up continuous compliance monitoring

**Assessment Criteria:**
- Zero trust architecture implementation (30%)
- Identity and access management (25%)
- Data protection and classification (20%)
- Network security controls (15%)
- Continuous monitoring (10%)

**Deliverables:**
- Zero trust architecture documentation
- IAM policy templates and guidelines
- Security assessment reports
- User access management procedures

---

## Topic 4: S3, Glacier, EBS, EFS, FSx

### **Practical Assignment 1: Multi-Tier Data Lifecycle Management**

**Case Study:**
DataVault Corp stores 500TB of customer data including active files, regulatory archives, and backup data. They need an automated data lifecycle management solution that optimizes costs while meeting regulatory requirements for data retention (7 years) and retrieval times (immediate for active, 24 hours for archives).

**Tasks to Complete:**
1. Design S3 bucket structure with appropriate storage classes
2. Implement automated lifecycle policies for cost optimization
3. Configure cross-region replication for disaster recovery
4. Set up S3 Intelligent-Tiering for automatic optimization
5. Implement Glacier vaults with different retrieval options
6. Configure S3 Event Notifications for processing workflows
7. Set up data encryption in transit and at rest
8. Create monitoring and cost reporting dashboards

**Assessment Criteria:**
- Storage architecture design (25%)
- Cost optimization effectiveness (25%)
- Data protection and compliance (20%)
- Automation implementation (15%)
- Monitoring and reporting (15%)

**Deliverables:**
- Storage architecture documentation
- Lifecycle policy configurations
- Cost analysis and projections
- Data recovery procedures

---

### **Practical Assignment 2: High-Performance Computing Storage**

**Case Study:**
GenomicsLab processes large genomic datasets requiring high-performance parallel access from 100+ compute instances. Workloads include real-time analysis requiring sub-millisecond latency and batch processing of multi-terabyte datasets. The solution must support POSIX compliance and scale to petabyte capacity.

**Tasks to Complete:**
1. Deploy FSx for Lustre with optimal performance configuration
2. Implement EFS with provisioned throughput mode
3. Configure EBS volumes with different performance tiers
4. Set up S3 integration for data staging and archival
5. Implement parallel data transfer optimization
6. Configure performance monitoring and alerting
7. Test various workload patterns and optimize
8. Create backup and disaster recovery strategies

**Assessment Criteria:**
- Performance optimization (30%)
- Scalability implementation (25%)
- Workload pattern analysis (20%)
- Cost vs. performance trade-offs (15%)
- Backup and recovery (10%)

**Deliverables:**
- Performance testing results
- Storage configuration documentation
- Workload optimization guide
- Disaster recovery plan

---

## Topic 5: Well-Architected Framework

### **Practical Assignment 1: Well-Architected Review and Optimization**

**Case Study:**
LegacyApp Inc. has been running their e-commerce platform on AWS for 3 years without following AWS best practices. They experience frequent outages, high costs, and security concerns. Conduct a comprehensive Well-Architected Review and implement improvements across all five pillars.

**Tasks to Complete:**
1. Conduct Well-Architected Framework assessment
2. Identify gaps across all five pillars
3. Prioritize improvements based on business impact
4. Implement operational excellence improvements
5. Enhance security posture and compliance
6. Improve reliability and fault tolerance
7. Optimize performance and user experience
8. Reduce costs through rightsizing and automation

**Assessment Criteria:**
- Comprehensive assessment quality (25%)
- Prioritization and business alignment (20%)
- Implementation effectiveness (25%)
- Cost optimization results (15%)
- Documentation and knowledge transfer (15%)

**Deliverables:**
- Well-Architected Review report
- Improvement implementation plan
- Before/after architecture comparison
- Cost optimization analysis

---

### **Practical Assignment 2: Green Cloud Architecture Design**

**Case Study:**
EcoTech Corp wants to build a new SaaS platform with sustainability as a core principle. Design an architecture that minimizes carbon footprint while maintaining high performance and availability. The solution should demonstrate best practices for environmental responsibility in cloud computing.

**Tasks to Complete:**
1. Design carbon-efficient architecture using renewable regions
2. Implement serverless and managed services for efficiency
3. Configure auto-scaling for optimal resource utilization
4. Use spot instances and reserved capacity strategically
5. Implement right-sizing recommendations automation
6. Configure sustainability monitoring and reporting
7. Optimize data transfer and storage for carbon efficiency
8. Create sustainability metrics dashboard

**Assessment Criteria:**
- Sustainability architecture design (30%)
- Resource optimization (25%)
- Performance vs. sustainability balance (20%)
- Monitoring and measurement (15%)
- Innovation and best practices (10%)

**Deliverables:**
- Sustainable architecture documentation
- Carbon footprint analysis
- Resource optimization strategies
- Sustainability metrics dashboard

---

## Topic 6: AWS CLI and SDKs

### **Practical Assignment 1: Infrastructure Automation with CLI**

**Case Study:**
DevOps team at StartupTech needs to automate their entire AWS infrastructure provisioning and management using AWS CLI. They manage multiple environments and need scripts that can deploy, update, and tear down infrastructure consistently while maintaining proper error handling and logging.

**Tasks to Complete:**
1. Create comprehensive CLI scripts for infrastructure deployment
2. Implement environment-specific configuration management
3. Add error handling and rollback capabilities
4. Create automated backup and restore procedures
5. Implement resource tagging and cost allocation
6. Add monitoring and alerting through CLI automation
7. Create CI/CD integration with CLI commands
8. Document all scripts with usage examples

**Assessment Criteria:**
- Script quality and error handling (30%)
- Automation completeness (25%)
- Documentation and usability (20%)
- CI/CD integration (15%)
- Best practices implementation (10%)

**Deliverables:**
- Automation script repository
- Environment configuration templates
- User documentation and tutorials
- CI/CD pipeline configurations

---

### **Practical Assignment 2: Python SDK Application Development**

**Case Study:**
DataProcessor Corp needs a Python application that integrates with multiple AWS services for their data processing pipeline. The application should handle S3 operations, DynamoDB interactions, SQS messaging, and CloudWatch monitoring while following AWS SDK best practices.

**Tasks to Complete:**
1. Develop Python application using Boto3 SDK
2. Implement proper error handling and retry logic
3. Configure credential management and security
4. Add comprehensive logging and monitoring
5. Implement asynchronous operations for performance
6. Create unit tests and integration tests
7. Add configuration management for different environments
8. Package application for deployment

**Assessment Criteria:**
- Application architecture and design (25%)
- Error handling and resilience (25%)
- Testing and quality assurance (20%)
- Performance optimization (15%)
- Security implementation (15%)

**Deliverables:**
- Python application source code
- Test suite and coverage reports
- Deployment documentation
- Performance benchmarking results

---

## Topic 7: AWS Organizations and Control Tower

### **Practical Assignment 1: Enterprise Account Management Platform**

**Case Study:**
GlobalCorp has 200+ AWS accounts across different business units and geographic regions. They need automated account provisioning, centralized billing management, security baseline enforcement, and compliance monitoring across all accounts.

**Tasks to Complete:**
1. Set up AWS Organizations with optimal OU structure
2. Implement AWS Control Tower for automated governance
3. Configure Service Control Policies for security enforcement
4. Set up automated account provisioning workflow
5. Implement centralized logging and monitoring
6. Configure cost allocation and budget management
7. Create compliance reporting automation
8. Set up cross-account role management

**Assessment Criteria:**
- Organizational structure design (25%)
- Automation implementation (25%)
- Security and compliance (25%)
- Cost management (15%)
- Operational efficiency (10%)

**Deliverables:**
- Account governance documentation
- Automation workflow diagrams
- Security baseline configurations
- Cost management reports

---

### **Practical Assignment 2: Multi-Account Security and Compliance**

**Case Study:**
RegulatedFirm operates in financial services and must comply with multiple regulations (SOX, PCI DSS, GDPR). They need a multi-account strategy that enforces compliance controls, maintains audit trails, and provides regulatory reporting across all business units.

**Tasks to Complete:**
1. Design compliant multi-account architecture
2. Implement preventive controls with SCPs
3. Configure detective controls with AWS Config
4. Set up automated remediation workflows
5. Create comprehensive audit trail collection
6. Implement data residency and sovereignty controls
7. Configure compliance reporting automation
8. Create incident response procedures

**Assessment Criteria:**
- Compliance architecture (30%)
- Preventive and detective controls (25%)
- Audit and reporting capabilities (20%)
- Automation quality (15%)
- Incident response procedures (10%)

**Deliverables:**
- Compliance architecture documentation
- Control implementation guides
- Audit reporting templates
- Incident response playbooks

---

## Topic 8: Multi-Account Strategy and Hybrid Cloud

### **Practical Assignment 1: Hybrid Cloud Identity Federation**

**Case Study:**
TechCorporation has 10,000 employees across multiple offices with existing Active Directory infrastructure. They need seamless access to AWS resources using corporate credentials while maintaining security and audit requirements for both on-premises and cloud resources.

**Tasks to Complete:**
1. Set up AWS SSO with Active Directory integration
2. Configure SAML federation for web console access
3. Implement programmatic access with STS
4. Set up cross-account role assumption
5. Configure fine-grained access controls
6. Implement session duration and MFA requirements
7. Set up audit logging for all access activities
8. Create user onboarding and offboarding automation

**Assessment Criteria:**
- Identity federation architecture (30%)
- Security implementation (25%)
- User experience and automation (20%)
- Audit and compliance (15%)
- Documentation and procedures (10%)

**Deliverables:**
- Identity federation documentation
- Access control policy templates
- User management procedures
- Security audit reports

---

### **Practical Assignment 2: Edge Computing and Hybrid Workloads**

**Case Study:**
ManufacturingTech operates 50 factory locations worldwide that generate IoT data requiring local processing for real-time decisions and cloud processing for analytics. Design a hybrid architecture that processes data locally while maintaining cloud connectivity for management and analytics.

**Tasks to Complete:**
1. Deploy AWS Outposts for local compute capacity
2. Configure AWS IoT Greengrass for edge processing
3. Set up hybrid networking with Direct Connect
4. Implement data synchronization strategies
5. Configure local and cloud-based analytics
6. Set up centralized monitoring and management
7. Implement edge security and device management
8. Create disaster recovery for edge locations

**Assessment Criteria:**
- Hybrid architecture design (30%)
- Edge computing implementation (25%)
- Data processing strategies (20%)
- Centralized management (15%)
- Security and disaster recovery (10%)

**Deliverables:**
- Hybrid architecture documentation
- Edge deployment procedures
- Data flow and processing diagrams
- Disaster recovery plans

---

## Topic 9: CloudWatch, CloudFormation, Systems Manager

### **Practical Assignment 1: Comprehensive Infrastructure Monitoring**

**Case Study:**
CloudOps team manages a complex multi-tier application with microservices, databases, and caching layers. They need comprehensive monitoring that provides application insights, infrastructure metrics, and business KPIs with automated alerting and remediation capabilities.

**Tasks to Complete:**
1. Implement custom CloudWatch metrics and dashboards
2. Configure CloudWatch Logs for centralized logging
3. Set up CloudWatch Alarms with multi-metric analysis
4. Create automated remediation with Systems Manager
5. Implement distributed tracing with X-Ray
6. Configure synthetic monitoring for user experience
7. Set up cost and performance optimization alerts
8. Create runbook automation for common issues

**Assessment Criteria:**
- Monitoring comprehensiveness (30%)
- Automation and remediation (25%)
- Dashboard design and usability (20%)
- Alert accuracy and relevance (15%)
- Documentation and procedures (10%)

**Deliverables:**
- Monitoring implementation documentation
- Custom dashboard configurations
- Automated remediation scripts
- Operational runbooks

---

### **Practical Assignment 2: Infrastructure as Code with CloudFormation**

**Case Study:**
ScaleUp company needs to deploy identical infrastructure stacks across multiple regions and environments. Create a comprehensive CloudFormation solution that supports parameterization, modularity, and lifecycle management for their entire infrastructure stack.

**Tasks to Complete:**
1. Design modular CloudFormation template architecture
2. Implement nested stacks for reusability
3. Create parameter and mapping strategies
4. Implement conditional resource deployment
5. Configure stack policies and drift detection
6. Set up automated testing and validation
7. Create CI/CD integration for infrastructure deployment
8. Implement disaster recovery automation

**Assessment Criteria:**
- Template architecture and modularity (30%)
- Parameterization and reusability (25%)
- Testing and validation (20%)
- CI/CD integration (15%)
- Documentation and best practices (10%)

**Deliverables:**
- CloudFormation template library
- Parameter configuration guides
- Testing framework documentation
- Deployment automation procedures

---

## Topic 10: FinOps and Cost Management

### **Practical Assignment 1: Enterprise Cost Optimization Platform**

**Case Study:**
MegaCorp spends $10M annually on AWS across 100+ accounts. Implement a comprehensive FinOps platform that provides cost visibility, optimization recommendations, budget controls, and chargeback capabilities to business units.

**Tasks to Complete:**
1. Implement comprehensive tagging strategy
2. Set up Cost Explorer with custom reports
3. Configure AWS Budgets with automated actions
4. Create cost allocation and chargeback system
5. Implement rightsizing recommendation automation
6. Set up Reserved Instance and Savings Plans optimization
7. Create executive dashboards and reporting
8. Implement cost anomaly detection and alerting

**Assessment Criteria:**
- Cost visibility implementation (25%)
- Optimization automation (25%)
- Chargeback accuracy (20%)
- Executive reporting (15%)
- Anomaly detection (15%)

**Deliverables:**
- FinOps platform documentation
- Cost optimization reports
- Chargeback methodology
- Executive dashboard configurations

---

### **Practical Assignment 2: Cloud Economics and Procurement Strategy**

**Case Study:**
GrowthTech is planning a 3-year cloud migration and needs a comprehensive procurement strategy that optimizes costs while supporting business growth. Develop a financial model that includes Reserved Instances, Savings Plans, and Spot Instance strategies.

**Tasks to Complete:**
1. Analyze current usage patterns and forecast growth
2. Develop Reserved Instance procurement strategy
3. Model Savings Plans scenarios and recommendations
4. Implement Spot Instance automation for appropriate workloads
5. Create financial modeling for different growth scenarios
6. Implement cost governance and approval workflows
7. Set up continuous optimization monitoring
8. Create procurement recommendation engine

**Assessment Criteria:**
- Financial modeling accuracy (30%)
- Procurement strategy effectiveness (25%)
- Growth scenario planning (20%)
- Automation implementation (15%)
- Governance framework (10%)

**Deliverables:**
- Financial models and projections
- Procurement strategy documentation
- Cost governance procedures
- Optimization automation scripts

---

## Topic 11: Kubernetes Essentials

### **Practical Assignment 1: Production Kubernetes Cluster Management**

**Case Study:**
ContainerCorp needs to deploy and manage a production-ready Kubernetes cluster on EKS that supports multiple applications with different scaling requirements, security policies, and resource needs. The cluster must support both stateful and stateless workloads.

**Tasks to Complete:**
1. Deploy EKS cluster with managed node groups
2. Configure RBAC and Pod Security Policies
3. Implement Horizontal and Vertical Pod Autoscaling
4. Set up Ingress controllers and load balancing
5. Configure persistent storage with EBS CSI driver
6. Implement monitoring with Prometheus and Grafana
7. Set up logging aggregation with Fluentd
8. Create backup and disaster recovery procedures

**Assessment Criteria:**
- Cluster architecture and security (30%)
- Scaling and performance (25%)
- Monitoring and observability (20%)
- Storage and persistence (15%)
- Operational procedures (10%)

**Deliverables:**
- EKS cluster configuration documentation
- Application deployment manifests
- Monitoring dashboard configurations
- Operational runbooks

---

### **Practical Assignment 2: Microservices Deployment and Service Mesh**

**Case Study:**
MicroTech has 15 microservices that need secure communication, traffic management, and observability. Implement a service mesh solution that provides end-to-end encryption, traffic routing, and comprehensive observability for all service interactions.

**Tasks to Complete:**
1. Deploy Istio service mesh on EKS cluster
2. Configure mTLS for service-to-service communication
3. Implement traffic management and routing policies
4. Set up circuit breakers and retry policies
5. Configure distributed tracing and metrics collection
6. Implement canary deployments with traffic splitting
7. Set up security policies and access controls
8. Create troubleshooting and debugging procedures

**Assessment Criteria:**
- Service mesh implementation (30%)
- Security and communication (25%)
- Traffic management (20%)
- Observability and debugging (15%)
- Operational complexity management (10%)

**Deliverables:**
- Service mesh configuration documentation
- Traffic management policies
- Security implementation guide
- Troubleshooting procedures

---

## Assessment Instructions and Grading Rubric

### **General Assessment Guidelines:**

**Time Allocation:**
- Assignment completion: 40 hours per assignment
- Documentation: 8 hours per assignment
- Presentation: 2 hours per assignment

**Submission Requirements:**
1. Complete working implementation
2. Infrastructure as Code templates
3. Architecture documentation with diagrams
4. Cost analysis and optimization report
5. Security assessment and recommendations
6. Operational procedures and runbooks
7. Video demonstration (15 minutes)
8. Peer review of another student's work

**Grading Scale:**
- A (90-100%): Exceptional implementation with innovation
- B (80-89%): Complete implementation meeting all requirements
- C (70-79%): Adequate implementation with minor gaps
- D (60-69%): Basic implementation with significant gaps
- F (<60%): Incomplete or non-functional implementation

**Evaluation Criteria:**
1. **Technical Implementation (40%)**
   - Functionality and correctness
   - Best practices adherence
   - Performance optimization
   - Security implementation

2. **Architecture and Design (25%)**
   - System design quality
   - Scalability considerations
   - Cost optimization
   - Innovation and creativity

3. **Documentation (20%)**
   - Completeness and accuracy
   - Clarity and organization
   - Operational procedures
   - Architecture diagrams

4. **Professional Skills (15%)**
   - Project management
   - Communication effectiveness
   - Peer collaboration
   - Problem-solving approach

**Note:** Each assignment builds upon previous knowledge and requires integration of multiple AWS services. Students should demonstrate not just technical skills but also business understanding and architectural thinking.